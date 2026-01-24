from __future__ import annotations
import yaml
import shlex
from pathlib import Path
from typing import List, Dict, Any, Mapping


def materialize(template_name: str, goal_title: str, vars: Mapping[str, str] | None = None) -> List[Dict[str, Any]]:
    p = Path(__file__).resolve().parent / "templates" / f"{template_name}.yaml"
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    vars = vars or {}

    def _subst(val: Any) -> Any:
        if isinstance(val, str):
            for k, v in vars.items():
                val = val.replace(f"${{{k}}}", str(v))
            return val
        if isinstance(val, list):
            out: List[str] = []
            for item in val:
                s = _subst(item)
                if isinstance(s, str):
                    out.extend(shlex.split(s))
                else:
                    out.append(s)
            return out
        return val

    steps = []
    for i, s in enumerate(data.get("steps", [])):
        cmd = _subst(s.get("cmd", []))
        steps.append({
            "idx": i,
            "name": s["name"],
            "cmd": cmd,
            "budget_s": float(s.get("budget_s", 60)),
        })
    return steps
