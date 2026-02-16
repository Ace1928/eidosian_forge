from __future__ import annotations

from typing import Any, Mapping


def coherence_from_workspace_summary(summary: Mapping[str, Any]) -> dict[str, float]:
    windows = int(summary.get("window_count") or 0)
    ignitions = int(summary.get("ignition_count") or 0)
    coherence_ratio = float(summary.get("coherence_ratio") or 0.0)
    ignition_density = (ignitions / windows) if windows else 0.0
    return {
        "coherence_ratio": round(coherence_ratio, 6),
        "ignition_density": round(float(ignition_density), 6),
    }
