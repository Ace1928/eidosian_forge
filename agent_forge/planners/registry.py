from __future__ import annotations
from core.contracts import Goal


def choose(goal: Goal) -> tuple[str, dict]:
    title = goal.title.lower()
    if "lint" in title:
        return "htn", {"template": "lint"}
    if "hygiene" in title:
        return "htn", {"template": "hygiene", "vars": {"pytest_args": "-q"}}
    return "htn", {"template": "hygiene", "vars": {"pytest_args": "-q"}}
