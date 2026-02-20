from __future__ import annotations

from pathlib import Path

from agent_forge.consciousness.kernel import ConsciousnessKernel
from agent_forge.core import events


class FlakyModule:
    name = "flaky_module"

    def __init__(self, fail_calls: int = 2) -> None:
        self._fail_calls = max(0, int(fail_calls))
        self.calls = 0

    def tick(self, ctx) -> None:
        self.calls += 1
        if self.calls <= self._fail_calls:
            raise RuntimeError(f"flaky-failure-{self.calls}")
        ctx.emit_event("flaky.ok", {"call": self.calls}, tags=["test", "watchdog"])


class LargePayloadModule:
    name = "large_payload_module"

    def tick(self, ctx) -> None:
        huge_text = "x" * 4096
        huge_items = list(range(256))
        ctx.emit_event(
            "large.event",
            {
                "text": huge_text,
                "items": huge_items,
                "nested": {"text": huge_text, "items": huge_items},
            },
            tags=["test", "payload"],
        )
        ctx.broadcast(
            "large_payload_module",
            {
                "kind": "PERCEPT",
                "confidence": 0.93,
                "salience": 0.87,
                "content": {
                    "text": huge_text,
                    "items": huge_items,
                },
            },
            tags=["test", "payload"],
        )


def test_kernel_watchdog_quarantines_and_recovers(tmp_path: Path) -> None:
    base = tmp_path / "state"
    flaky = FlakyModule(fail_calls=2)
    kernel = ConsciousnessKernel(
        base,
        modules=[flaky],
        config={
            "kernel_watchdog_enabled": True,
            "kernel_watchdog_max_consecutive_errors": 2,
            "kernel_watchdog_quarantine_beats": 2,
        },
        seed=11,
    )

    for _ in range(6):
        kernel.tick()

    all_events = events.iter_events(base, limit=None)
    module_errors = [e for e in all_events if str(e.get("type")) == "consciousness.module_error"]
    quarantined = [e for e in all_events if str(e.get("type")) == "consciousness.module_quarantined"]
    recovered = [e for e in all_events if str(e.get("type")) == "consciousness.module_recovered"]
    ok_events = [e for e in all_events if str(e.get("type")) == "flaky.ok"]

    assert flaky.calls == 4
    assert len(module_errors) == 2
    assert len(quarantined) == 1
    assert len(recovered) == 1
    assert len(ok_events) == 2

    quarantine_data = quarantined[0].get("data") if isinstance(quarantined[0].get("data"), dict) else {}
    assert int(quarantine_data.get("quarantine_beats") or 0) == 2
    assert int(quarantine_data.get("threshold") or 0) == 2
    assert int(quarantine_data.get("quarantined_until_beat") or 0) >= 4


def test_payload_safety_truncates_event_and_broadcast_payloads(tmp_path: Path) -> None:
    base = tmp_path / "state"
    kernel = ConsciousnessKernel(
        base,
        modules=[LargePayloadModule()],
        config={
            "consciousness_max_payload_bytes": 1024,
            "consciousness_max_depth": 5,
            "consciousness_max_collection_items": 16,
            "consciousness_max_string_chars": 80,
            "consciousness_payload_truncation_event": True,
        },
        seed=7,
    )
    kernel.tick()

    all_events = events.iter_events(base, limit=None)
    large_evt = next(e for e in all_events if str(e.get("type")) == "large.event")
    large_data = large_evt.get("data") if isinstance(large_evt.get("data"), dict) else {}
    text_value = str(large_data.get("text") or "")
    items_value = large_data.get("items") if isinstance(large_data.get("items"), list) else []
    assert len(text_value) <= 96
    assert len(items_value) <= 17

    broadcast_evt = next(
        e
        for e in all_events
        if str(e.get("type")) == "workspace.broadcast"
        and isinstance(e.get("data"), dict)
        and isinstance((e.get("data") or {}).get("payload"), dict)
        and str(((e.get("data") or {}).get("payload") or {}).get("source_module")) == "large_payload_module"
    )
    payload = (
        (broadcast_evt.get("data") or {}).get("payload")
        if isinstance((broadcast_evt.get("data") or {}).get("payload"), dict)
        else {}
    )
    content = payload.get("content") if isinstance(payload.get("content"), dict) else {}
    b_text = str(content.get("text") or "")
    b_items = content.get("items") if isinstance(content.get("items"), list) else []
    assert len(b_text) <= 96
    assert len(b_items) <= 17

    trunc_events = [e for e in all_events if str(e.get("type")) == "consciousness.payload_truncated"]
    assert len(trunc_events) >= 2
    source_types = {
        str((e.get("data") or {}).get("source_type")) for e in trunc_events if isinstance(e.get("data"), dict)
    }
    assert {"event", "broadcast"}.issubset(source_types)
