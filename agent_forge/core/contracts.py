from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional, Dict, Any

Id = str  # ULIDs later


@dataclass
class Goal:
    id: Id
    title: str
    drive: str
    created_at: str


@dataclass
class Plan:
    id: Id
    goal_id: Id
    kind: Literal["htn", "mcts", "adhoc"]
    meta: Dict[str, Any]
    created_at: str


@dataclass
class Step:
    id: Id
    plan_id: Id
    idx: int
    name: str
    cmd: str
    budget_s: float
    status: Literal["todo", "running", "ok", "fail"]
    created_at: str


@dataclass
class Run:
    id: Id
    step_id: Id
    started_at: str
    ended_at: Optional[str]
    rc: Optional[int]
    bytes_out: int
    notes: str
