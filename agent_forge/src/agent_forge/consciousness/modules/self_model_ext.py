from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from ..metrics.agency import agency_confidence
from ..types import TickContext, WorkspacePayload, normalize_workspace_payload


def _event_data(evt: Mapping[str, Any]) -> Mapping[str, Any]:
    return evt.get("data") if isinstance(evt.get("data"), Mapping) else {}


def _latest_event(events: list[Mapping[str, Any]], etype: str) -> Optional[Mapping[str, Any]]:
    for evt in reversed(events):
        if str(evt.get("type") or "") == etype:
            return evt
    return None


class SelfModelExtModule:
    name = "self_model_ext"

    def __init__(self) -> None:
        self._last_agency: Optional[float] = None
        self._last_boundary: Optional[float] = None

    def _observed_signature(self, events: list[Mapping[str, Any]], expected: Mapping[str, Any]) -> Dict[str, Any]:
        expected_event_type = str(expected.get("expected_event_type") or "")
        expected_source = str(expected.get("expected_source") or "")
        expected_kind = str(expected.get("expected_kind") or "")

        observed_event_type = ""
        observed_source = ""
        observed_kind = ""

        for evt in reversed(events):
            etype = str(evt.get("type") or "")
            data = _event_data(evt)
            source = str(data.get("source") or "")
            payload = data.get("payload") if isinstance(data.get("payload"), Mapping) else {}
            kind = str(payload.get("kind") or "")
            if expected_event_type and etype != expected_event_type:
                continue
            if expected_source and source and source != expected_source:
                continue
            observed_event_type = etype
            observed_source = source
            observed_kind = kind
            break

        return {
            "expected_event_type": expected_event_type,
            "expected_source": expected_source,
            "expected_kind": expected_kind,
            "observed_event_type": observed_event_type,
            "observed_source": observed_source,
            "observed_kind": observed_kind,
        }

    def _boundary_stability(self, events: list[Mapping[str, Any]]) -> tuple[float, Dict[str, Any]]:
        actions = [evt for evt in events if str(evt.get("type") or "") == "policy.action"]
        efferences = [evt for evt in events if str(evt.get("type") or "") == "policy.efference"]
        if not actions:
            return 0.0, {"actions": 0, "efferences": len(efferences), "matched": 0}

        action_ids = {
            str((_event_data(evt)).get("action_id") or "")
            for evt in actions
            if str((_event_data(evt)).get("action_id") or "")
        }
        eff_ids = {
            str((_event_data(evt)).get("action_id") or "")
            for evt in efferences
            if str((_event_data(evt)).get("action_id") or "")
        }
        matched = len(action_ids & eff_ids)
        stability = matched / max(len(action_ids), 1)
        return round(stability, 6), {
            "actions": len(action_ids),
            "efferences": len(eff_ids),
            "matched": matched,
        }

    def tick(self, ctx: TickContext) -> None:
        window = int(ctx.config.get("self_observation_window", 120))
        emit_delta = float(ctx.config.get("self_emit_delta_threshold", 0.05))
        perturbations = ctx.perturbations_for(self.name)
        if any(str(p.get("kind") or "") == "drop" for p in perturbations):
            return
        if any(str(p.get("kind") or "") == "delay" for p in perturbations) and (ctx.beat_count % 2 == 1):
            return
        noise_mag = max(
            [
                max(0.0, min(1.0, float(p.get("magnitude") or 0.0)))
                for p in perturbations
                if str(p.get("kind") or "") == "noise"
            ]
            or [0.0]
        )
        clamp_mag = max(
            [
                max(0.0, min(1.0, float(p.get("magnitude") or 0.0)))
                for p in perturbations
                if str(p.get("kind") or "") == "clamp"
            ]
            or [0.0]
        )
        scramble = any(str(p.get("kind") or "") == "scramble" for p in perturbations)

        combined = list(ctx.recent_events)[-window:] + list(ctx.emitted_events)

        latest_efference = _latest_event(combined, "policy.efference")
        if latest_efference is None:
            return

        eff_data = _event_data(latest_efference)
        predicted = (
            eff_data.get("predicted_observation") if isinstance(eff_data.get("predicted_observation"), Mapping) else {}
        )
        observed_sig = self._observed_signature(combined, predicted)
        predicted_sig = {
            "expected_event_type": str(predicted.get("expected_event_type") or ""),
            "expected_source": str(predicted.get("expected_source") or ""),
            "expected_kind": str(predicted.get("expected_kind") or ""),
        }
        observed_for_match = {
            "expected_event_type": observed_sig.get("observed_event_type"),
            "expected_source": observed_sig.get("observed_source"),
            "expected_kind": observed_sig.get("observed_kind"),
        }
        if scramble:
            observed_for_match["expected_event_type"] = predicted_sig.get("expected_event_type")
            observed_for_match["expected_source"] = predicted_sig.get("expected_source")
            observed_for_match["expected_kind"] = ""
        agency = agency_confidence(predicted_sig, observed_for_match)

        boundary, boundary_meta = self._boundary_stability(combined)
        if noise_mag > 0.0:
            agency = max(0.0, min(1.0, agency + ctx.rng.uniform(-noise_mag, noise_mag)))
            boundary = max(0.0, min(1.0, boundary + ctx.rng.uniform(-noise_mag, noise_mag)))
        if clamp_mag > 0.0:
            cap = max(0.0, 1.0 - clamp_mag)
            agency = min(agency, cap)
            boundary = min(boundary, cap)
        agency = round(float(agency), 6)
        boundary = round(float(boundary), 6)

        agency_evt = ctx.emit_event(
            "self.agency_estimate",
            {
                "action_id": eff_data.get("action_id"),
                "agency_confidence": agency,
                "predicted": predicted_sig,
                "observed": observed_sig,
            },
            tags=["consciousness", "self", "agency"],
            corr_id=latest_efference.get("corr_id"),
            parent_id=latest_efference.get("parent_id"),
        )

        ctx.emit_event(
            "self.boundary_estimate",
            {
                "boundary_stability": boundary,
                "control_graph": {
                    "policy_action_to_efference_match": boundary,
                    "meta": boundary_meta,
                },
            },
            tags=["consciousness", "self", "boundary"],
            corr_id=agency_evt.get("corr_id"),
            parent_id=agency_evt.get("parent_id"),
        )

        ctx.metric("consciousness.agency", agency)
        ctx.metric("consciousness.boundary_stability", boundary)

        should_emit = False
        if self._last_agency is None or self._last_boundary is None:
            should_emit = True
        else:
            if abs(agency - self._last_agency) >= emit_delta:
                should_emit = True
            if abs(boundary - self._last_boundary) >= emit_delta:
                should_emit = True

        self._last_agency = agency
        self._last_boundary = boundary

        if should_emit:
            payload = WorkspacePayload(
                kind="SELF",
                source_module="self_model_ext",
                content={
                    "agency_confidence": agency,
                    "boundary_stability": boundary,
                    "boundary_meta": boundary_meta,
                    "action_id": eff_data.get("action_id"),
                },
                confidence=agency,
                salience=max(agency, boundary),
                links={
                    "corr_id": agency_evt.get("corr_id"),
                    "parent_id": agency_evt.get("parent_id"),
                    "memory_ids": [],
                },
            ).as_dict()
            payload = normalize_workspace_payload(payload, fallback_kind="SELF", source_module="self_model_ext")
            ctx.broadcast(
                "self_model_ext",
                payload,
                tags=["consciousness", "self", "broadcast"],
                corr_id=agency_evt.get("corr_id"),
                parent_id=agency_evt.get("parent_id"),
            )
