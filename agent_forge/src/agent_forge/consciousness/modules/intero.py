from __future__ import annotations

from typing import Any, Mapping

from agent_forge.core import workspace

from ..types import TickContext
from ..types import WorkspacePayload, clamp01, normalize_workspace_payload


def _event_data(evt: Mapping[str, Any]) -> Mapping[str, Any]:
    return evt.get("data") if isinstance(evt.get("data"), Mapping) else {}


def _latest_metric(events: list[Mapping[str, Any]], key: str) -> float | None:
    for evt in reversed(events):
        if str(evt.get("type") or "") != "metrics.sample":
            continue
        data = _event_data(evt)
        if str(data.get("key") or "") != key:
            continue
        try:
            return float(data.get("value"))
        except (TypeError, ValueError):
            return None
    return None


class InteroModule:
    name = "intero"

    def tick(self, ctx: TickContext) -> None:
        alpha = clamp01(ctx.config.get("intero_drive_alpha"), default=0.22)
        emit_threshold = clamp01(
            ctx.config.get("intero_broadcast_threshold"), default=0.45
        )

        state = ctx.module_state(
            self.name,
            defaults={
                "drives": {
                    "energy": 0.5,
                    "threat": 0.2,
                    "curiosity": 0.4,
                    "coherence_hunger": 0.3,
                    "novelty_hunger": 0.35,
                },
                "integral": {},
            },
        )
        drives = state.get("drives") if isinstance(state.get("drives"), Mapping) else {}
        prev = {str(k): clamp01(v, default=0.5) for k, v in drives.items()}
        integral_map = (
            state.get("integral") if isinstance(state.get("integral"), Mapping) else {}
        )
        integral = {
            str(k): float(v)
            for k, v in integral_map.items()
            if isinstance(v, (int, float))
        }

        events = ctx.all_events()
        recent = events[-260:]
        ws = workspace.summary(
            ctx.state_dir, limit=220, window_seconds=1.0, min_sources=3
        )
        coherence = float(ws.get("coherence_ratio") or 0.0)
        ignition_count = float(ws.get("ignition_count") or 0.0)
        window_count = float(ws.get("window_count") or 1.0)
        ignition_ratio = clamp01(ignition_count / max(window_count, 1.0), default=0.0)

        pred_error = _latest_metric(recent, "consciousness.world.prediction_error")
        if pred_error is None:
            for evt in reversed(recent):
                if str(evt.get("type") or "") == "world.prediction_error":
                    data = _event_data(evt)
                    try:
                        pred_error = float(data.get("prediction_error"))
                    except (TypeError, ValueError):
                        pred_error = None
                    break
        pred_error = clamp01(pred_error, default=0.4)

        module_errors = sum(
            1
            for evt in recent
            if str(evt.get("type") or "") == "consciousness.module_error"
        )
        module_error_rate = clamp01(module_errors / max(len(recent), 1), default=0.0)

        sense_novelties: list[float] = []
        for evt in recent:
            if str(evt.get("type") or "") != "sense.percept":
                continue
            data = _event_data(evt)
            sense_novelties.append(clamp01(data.get("novelty"), default=0.4))
        avg_novelty = (
            sum(sense_novelties[-24:]) / max(len(sense_novelties[-24:]), 1)
            if sense_novelties
            else 0.35
        )

        targets = {
            "energy": clamp01(
                1.0 - (0.65 * pred_error) - (0.8 * module_error_rate), default=0.5
            ),
            "threat": clamp01(
                (0.65 * pred_error)
                + (0.9 * module_error_rate)
                + (0.2 * (1.0 - coherence)),
                default=0.2,
            ),
            "curiosity": clamp01(
                (0.55 * pred_error)
                + (0.35 * (1.0 - ignition_ratio))
                + (0.2 * avg_novelty),
                default=0.4,
            ),
            "coherence_hunger": clamp01(max(0.0, 0.55 - coherence) * 1.6, default=0.3),
            "novelty_hunger": clamp01(
                (0.5 * avg_novelty)
                + (0.35 * (1.0 - ignition_ratio))
                + (0.2 * pred_error),
                default=0.35,
            ),
        }

        updated: dict[str, float] = {}
        strongest_name = ""
        strongest_strength = 0.0
        for drive_name, target in targets.items():
            current = prev.get(drive_name, 0.5)
            value = (1.0 - alpha) * current + (alpha * target)
            value = clamp01(value, default=current)
            err = target - value
            integral[drive_name] = float(integral.get(drive_name, 0.0)) + err
            updated[drive_name] = round(value, 6)
            strength = abs(err)
            if strength > strongest_strength:
                strongest_name = drive_name
                strongest_strength = strength

            drive_evt = ctx.emit_event(
                "intero.drive",
                {
                    "drive_name": drive_name,
                    "value": round(value, 6),
                    "target": round(target, 6),
                    "setpoint_error": round(err, 6),
                    "integral": round(float(integral.get(drive_name, 0.0)), 6),
                    "strength": round(abs(err), 6),
                    "signals": {
                        "coherence": round(coherence, 6),
                        "prediction_error": round(pred_error, 6),
                        "module_error_rate": round(module_error_rate, 6),
                        "avg_novelty": round(avg_novelty, 6),
                    },
                },
                tags=["consciousness", "intero", "drive"],
            )
            ctx.metric(f"consciousness.intero.{drive_name}", float(updated[drive_name]))

            if abs(err) >= emit_threshold:
                payload = WorkspacePayload(
                    kind="DRIVE",
                    source_module=self.name,
                    content={
                        "drive_name": drive_name,
                        "value": round(value, 6),
                        "target": round(target, 6),
                        "setpoint_error": round(err, 6),
                    },
                    confidence=clamp01(1.0 - abs(err), default=0.5),
                    salience=clamp01(abs(err), default=0.5),
                    links={
                        "corr_id": drive_evt.get("corr_id"),
                        "parent_id": drive_evt.get("parent_id"),
                        "memory_ids": [],
                    },
                ).as_dict()
                payload = normalize_workspace_payload(
                    payload, fallback_kind="DRIVE", source_module=self.name
                )
                ctx.broadcast(
                    self.name,
                    payload,
                    tags=["consciousness", "intero", "broadcast"],
                    corr_id=drive_evt.get("corr_id"),
                    parent_id=drive_evt.get("parent_id"),
                )

        state["drives"] = updated
        state["integral"] = {k: round(v, 6) for k, v in integral.items()}
        state["signals"] = {
            "coherence": round(coherence, 6),
            "prediction_error": round(pred_error, 6),
            "module_error_rate": round(module_error_rate, 6),
            "avg_novelty": round(avg_novelty, 6),
            "ignition_ratio": round(ignition_ratio, 6),
        }
        state["strongest"] = {
            "drive": strongest_name,
            "strength": round(float(strongest_strength), 6),
        }
