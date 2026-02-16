from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping

from ..types import TickContext, WorkspacePayload, clamp01, normalize_workspace_payload


def _parse_iso(ts: str) -> datetime | None:
    try:
        text = ts
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


def _now_iso(now: datetime) -> str:
    return now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _item_key(payload: Mapping[str, Any]) -> str:
    content = (
        payload.get("content") if isinstance(payload.get("content"), Mapping) else {}
    )
    for key in ("candidate_id", "report_id", "action_id"):
        value = str(content.get(key) or "")
        if value:
            return value
    return str(payload.get("id") or "")


def _related_signature(payload: Mapping[str, Any]) -> str:
    content = (
        payload.get("content") if isinstance(payload.get("content"), Mapping) else {}
    )
    return (
        f"{content.get('source_module','')}|"
        f"{content.get('source_event_type','')}|"
        f"{payload.get('kind','')}"
    )


class WorkingSetModule:
    name = "working_set"

    def tick(self, ctx: TickContext) -> None:
        capacity = max(1, int(ctx.config.get("working_set_capacity", 7)))
        half_life = max(
            0.1, float(ctx.config.get("working_set_decay_half_life_secs", 8.0))
        )
        min_salience = clamp01(ctx.config.get("working_set_min_salience"), default=0.08)
        emit_interval = max(
            0.2, float(ctx.config.get("working_set_emit_interval_secs", 2.0))
        )
        scan_limit = max(10, int(ctx.config.get("working_set_scan_broadcasts", 120)))
        perturbations = ctx.perturbations_for(self.name)
        if any(str(p.get("kind") or "") == "drop" for p in perturbations):
            ctx.metric("consciousness.working_set.size", 0.0)
            return
        if any(str(p.get("kind") or "") == "delay" for p in perturbations) and (ctx.beat_count % 2 == 1):
            return
        noise_mag = max(
            [
                clamp01(p.get("magnitude"), default=0.0)
                for p in perturbations
                if str(p.get("kind") or "") == "noise"
            ]
            or [0.0]
        )
        clamp_cap = max(
            [
                clamp01(p.get("magnitude"), default=0.0)
                for p in perturbations
                if str(p.get("kind") or "") == "clamp"
            ]
            or [0.0]
        )
        scramble = any(str(p.get("kind") or "") == "scramble" for p in perturbations)
        if clamp_cap > 0.0:
            capacity = max(1, int(round(capacity * max(0.15, 1.0 - clamp_cap))))

        state = ctx.module_state(
            self.name,
            defaults={
                "active_items": [],
                "last_update_ts": _now_iso(ctx.now),
                "last_emit_ts": "",
            },
        )
        previous_items = list(state.get("active_items") or [])
        last_update = _parse_iso(str(state.get("last_update_ts") or ""))
        elapsed = 0.0
        if last_update is not None:
            elapsed = max(0.0, (ctx.now - last_update).total_seconds())

        active: list[Dict[str, Any]] = []
        decay_factor = 0.5 ** (elapsed / half_life) if elapsed > 0.0 else 1.0
        for row in previous_items:
            if not isinstance(row, Mapping):
                continue
            salience = clamp01(row.get("salience"), default=0.0) * decay_factor
            if salience < min_salience:
                continue
            item = dict(row)
            item["salience"] = round(salience, 6)
            if noise_mag > 0.0:
                item["salience"] = round(
                    clamp01(item["salience"] + ctx.rng.uniform(-noise_mag, noise_mag), default=item["salience"]),
                    6,
                )
            active.append(item)

        by_key: Dict[str, Dict[str, Any]] = {
            str(item.get("item_id") or item.get("id") or ""): item
            for item in active
            if str(item.get("item_id") or item.get("id") or "")
        }

        changed = False
        recent = ctx.latest_events("workspace.broadcast", k=scan_limit)
        for evt in recent:
            data = evt.get("data") if isinstance(evt.get("data"), Mapping) else {}
            payload = (
                data.get("payload") if isinstance(data.get("payload"), Mapping) else {}
            )
            if not payload:
                continue
            kind = str(payload.get("kind") or "")
            if kind not in {"GW_WINNER", "PRED_ERR", "PLAN", "SELF", "META", "REPORT"}:
                continue
            item_id = _item_key(payload)
            if not item_id:
                continue

            salience = clamp01(payload.get("salience"), default=0.5)
            if noise_mag > 0.0:
                salience = clamp01(salience + ctx.rng.uniform(-noise_mag, noise_mag), default=salience)
            confidence = clamp01(payload.get("confidence"), default=0.5)
            links = (
                payload.get("links")
                if isinstance(payload.get("links"), Mapping)
                else {}
            )
            existing = by_key.get(item_id)
            if existing:
                old_salience = clamp01(existing.get("salience"), default=0.0)
                new_salience = max(old_salience, salience)
                if abs(new_salience - old_salience) > 1e-9:
                    existing["salience"] = round(new_salience, 6)
                    changed = True
                existing["last_seen_ts"] = str(evt.get("ts") or _now_iso(ctx.now))
                continue

            by_key[item_id] = {
                "item_id": item_id,
                "kind": kind,
                "salience": round(salience, 6),
                "confidence": round(confidence, 6),
                "signature": _related_signature(payload),
                "links": dict(links),
                "entered_ts": str(evt.get("ts") or _now_iso(ctx.now)),
                "last_seen_ts": str(evt.get("ts") or _now_iso(ctx.now)),
                "source": str(
                    data.get("source") or payload.get("source_module") or "unknown"
                ),
            }
            changed = True

        ranked = sorted(
            by_key.values(),
            key=lambda item: (
                clamp01(item.get("salience"), default=0.0),
                str(item.get("last_seen_ts") or ""),
            ),
            reverse=True,
        )
        if scramble and len(ranked) > 1:
            ctx.rng.shuffle(ranked)
        trimmed = ranked[:capacity]
        if len(trimmed) != len(previous_items):
            changed = True

        state["active_items"] = trimmed
        state["last_update_ts"] = _now_iso(ctx.now)

        size = float(len(trimmed))
        ctx.metric("consciousness.working_set.size", size)

        last_emit = _parse_iso(str(state.get("last_emit_ts") or ""))
        since_emit = (ctx.now - last_emit).total_seconds() if last_emit else 10_000.0
        should_emit = changed or since_emit >= emit_interval
        if not should_emit:
            return

        content = {
            "size": len(trimmed),
            "capacity": capacity,
            "items": [
                {
                    "item_id": item.get("item_id"),
                    "kind": item.get("kind"),
                    "source": item.get("source"),
                    "salience": item.get("salience"),
                    "confidence": item.get("confidence"),
                }
                for item in trimmed
            ],
        }
        update_evt = ctx.emit_event(
            "wm.state",
            content,
            tags=["consciousness", "working_set", "state"],
        )
        payload = WorkspacePayload(
            kind="WM_STATE",
            source_module=self.name,
            content=content,
            confidence=0.8,
            salience=min(1.0, 0.2 + (0.1 * len(trimmed))),
            links={
                "corr_id": update_evt.get("corr_id"),
                "parent_id": update_evt.get("parent_id"),
                "memory_ids": [],
            },
        ).as_dict()
        payload = normalize_workspace_payload(
            payload, fallback_kind="WM_STATE", source_module=self.name
        )
        ctx.broadcast(
            self.name,
            payload,
            tags=["consciousness", "working_set", "broadcast"],
            corr_id=update_evt.get("corr_id"),
            parent_id=update_evt.get("parent_id"),
        )
        state["last_emit_ts"] = _now_iso(ctx.now)
