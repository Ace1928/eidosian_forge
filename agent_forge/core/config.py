#!/usr/bin/env python3
"""
Strict config loader for Eidos E3.

- Reads cfg/{self,drives,budgets,policies,skills}.yaml
- Validates schemas and types.
- Expands ${ENV} in string values.
- CLI:
    python -m core.config --dir cfg --print
    python -m core.config --dir cfg --validate
    python -m core.config --dir cfg --json
"""

from __future__ import annotations
import argparse, dataclasses as dc, json, os, re, sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

try:
    import yaml  # type: ignore
except Exception as e:  # pragma: no cover
    print("Missing dependency: PyYAML. Activate venv and run: pip install pyyaml", file=sys.stderr)
    sys.exit(2)

# ---------- dataclasses (local, no external deps) ----------

@dc.dataclass
class SelfCfg:
    name: str
    workspace: str
    axioms: List[str]
    non_negotiables: List[str]
    temperament: Mapping[str, float]
    commit_style: str

@dc.dataclass
class DriveSpec:
    metric: str
    target: Union[int, float]
    weight: float
    direction: str = "increase"  # "increase" or "decrease"

@dc.dataclass
class DrivesCfg:
    drives: Mapping[str, DriveSpec]
    activation: Mapping[str, float]

@dc.dataclass
class ScopeBudget:
    wall_clock: int
    cpu_time: int
    retry_limit: int

@dc.dataclass
class BudgetsCfg:
    global_: Mapping[str, Union[int, float]]
    scopes: Mapping[str, ScopeBudget]

@dc.dataclass
class AutonomyRung:
    name: str
    act: bool
    require_approvals: str
    min_competence: float
    max_risk: float

@dc.dataclass
class RiskModel:
    weights: Mapping[str, float]
    cutoffs: Mapping[str, float]

@dc.dataclass
class PoliciesCfg:
    autonomy_ladder: List[AutonomyRung]
    risk_model: RiskModel
    confidence: Mapping[str, float]
    approvals: Mapping[str, Any]

@dc.dataclass
class Skill:
    level: float
    brier: float

@dc.dataclass
class SkillsCfg:
    competencies: Mapping[str, Skill]

@dc.dataclass
class Config:
    self: SelfCfg
    drives: DrivesCfg
    budgets: BudgetsCfg
    policies: PoliciesCfg
    skills: SkillsCfg
    path: Path

# ---------- helpers ----------

_ENV_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

def _expand_env(obj: Any) -> Any:
    if isinstance(obj, str):
        def repl(m): return os.environ.get(m.group(1), m.group(0))
        return _ENV_VAR.sub(repl, obj)
    if isinstance(obj, list):  return [_expand_env(x) for x in obj]
    if isinstance(obj, dict):  return {k: _expand_env(v) for k, v in obj.items()}
    return obj

def _read_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return _expand_env(data)

def _require(d: Mapping[str, Any], key: str, typ, ctx: str):
    if key not in d:
        raise ValueError(f"Missing key '{key}' in {ctx}")
    val = d[key]
    if typ is not Any and not isinstance(val, typ):
        raise TypeError(f"Key '{key}' in {ctx} must be {typ}, got {type(val)}")
    return val

def _cast_map(d: Mapping[str, Any], key_types: Tuple[type, ...]=(str,)) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if not isinstance(k, key_types):
            raise TypeError(f"Non-string map key {k!r}")
        out[str(k)] = v
    return out

# ---------- builders (strict validation, friendly messages) ----------

def _build_self(d: Mapping[str, Any]) -> SelfCfg:
    return SelfCfg(
        name=_require(d, "name", str, "self.yaml"),
        workspace=_require(d, "workspace", str, "self.yaml"),
        axioms=list(_require(d, "axioms", list, "self.yaml")),
        non_negotiables=list(_require(d, "non_negotiables", list, "self.yaml")),
        temperament=_require(d, "temperament", dict, "self.yaml"),
        commit_style=_require(d, "commit_style", str, "self.yaml"),
    )

def _build_drives(d: Mapping[str, Any]) -> DrivesCfg:
    drives = _require(d, "drives", dict, "drives.yaml")
    spec_map: Dict[str, DriveSpec] = {}
    for name, cfg in _cast_map(drives).items():
        ctx = f"drives.yaml:drives[{name}]"
        metric = _require(cfg, "metric", str, ctx)
        target = _require(cfg, "target", (int, float), ctx)
        weight = _require(cfg, "weight", (int, float), ctx)
        direction = cfg.get("direction", "increase")
        if direction not in ("increase", "decrease"):
            raise ValueError(f"{ctx}.direction must be 'increase' or 'decrease'")
        spec_map[name] = DriveSpec(metric=metric, target=float(target), weight=float(weight), direction=direction)
    activation = _require(d, "activation", dict, "drives.yaml")
    return DrivesCfg(drives=spec_map, activation=activation)

def _build_budgets(d: Mapping[str, Any]) -> BudgetsCfg:
    g = _require(d, "global", dict, "budgets.yaml")
    scopes = _require(d, "scopes", dict, "budgets.yaml")
    scope_map: Dict[str, ScopeBudget] = {}
    for sname, scfg in _cast_map(scopes).items():
        ctx = f"budgets.yaml:scopes[{sname}]"
        scope_map[sname] = ScopeBudget(
            wall_clock=int(_require(scfg, "wall_clock", (int, float), ctx)),
            cpu_time=int(_require(scfg, "cpu_time", (int, float), ctx)),
            retry_limit=int(_require(scfg, "retry_limit", (int, float), ctx)),
        )
    return BudgetsCfg(global_=g, scopes=scope_map)

def _build_policies(d: Mapping[str, Any]) -> PoliciesCfg:
    ladder = _require(d, "autonomy_ladder", list, "policies.yaml")
    rungs: List[AutonomyRung] = []
    for i, rung in enumerate(ladder):
        ctx = f"policies.yaml:autonomy_ladder[{i}]"
        rungs.append(AutonomyRung(
            name=_require(rung, "name", str, ctx),
            act=bool(_require(rung, "act", (bool, int), ctx)),
            require_approvals=_require(rung, "require_approvals", str, ctx),
            min_competence=float(_require(rung, "min_competence", (int, float), ctx)),
            max_risk=float(_require(rung, "max_risk", (int, float), ctx)),
        ))
    rm = _require(d, "risk_model", dict, "policies.yaml")
    risk = RiskModel(
        weights=_require(rm, "weights", dict, "policies.yaml:risk_model"),
        cutoffs=_require(rm, "cutoffs", dict, "policies.yaml:risk_model"),
    )
    confidence = _require(d, "confidence", dict, "policies.yaml")
    approvals  = _require(d, "approvals", dict, "policies.yaml")
    return PoliciesCfg(autonomy_ladder=rungs, risk_model=risk, confidence=confidence, approvals=approvals)

def _build_skills(d: Mapping[str, Any]) -> SkillsCfg:
    comp = _require(d, "competencies", dict, "skills.yaml")
    cmap: Dict[str, Skill] = {}
    for k, v in _cast_map(comp).items():
        ctx = f"skills.yaml:competencies[{k}]"
        cmap[k] = Skill(level=float(_require(v, "level", (int, float), ctx)),
                        brier=float(_require(v, "brier", (int, float), ctx)))
    return SkillsCfg(competencies=cmap)

# ---------- public API ----------

def load_all(cfg_dir: Union[str, Path]) -> Config:
    base = Path(cfg_dir)
    if not base.is_dir():
        raise FileNotFoundError(f"Config directory not found: {base}")
    self_d   = _read_yaml(base / "self.yaml")
    drives_d = _read_yaml(base / "drives.yaml")
    budgets_d= _read_yaml(base / "budgets.yaml")
    policies_d=_read_yaml(base / "policies.yaml")
    skills_d = _read_yaml(base / "skills.yaml")

    cfg = Config(
        self=_build_self(self_d),
        drives=_build_drives(drives_d),
        budgets=_build_budgets(budgets_d),
        policies=_build_policies(policies_d),
        skills=_build_skills(skills_d),
        path=base,
    )
    _sanity(cfg)
    return cfg

def _sanity(cfg: Config) -> None:
    # basic sanity: weights sum approx <= 1.5; targets non-negative; ladders sorted by risk descending
    total_w = sum(ds.weight for ds in cfg.drives.drives.values())
    if total_w <= 0 or total_w > 2.0:
        raise ValueError(f"drives weights sum unreasonable: {total_w}")
    # autonomy ladder monotonicity checks
    r = cfg.policies.autonomy_ladder
    for i in range(1, len(r)):
        if r[i].min_competence < r[i-1].min_competence:
            raise ValueError("autonomy_ladder not sorted by increasing min_competence")

def to_dict(cfg: Config) -> Dict[str, Any]:
    def enc(o):
        if dc.is_dataclass(o):
            return {k: enc(v) for k, v in dc.asdict(o).items()}
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, list):
            return [enc(x) for x in o]
        if isinstance(o, dict):
            return {k: enc(v) for k, v in o.items()}
        return o
    return enc(cfg)

# ---------- CLI ----------

def _cli():
    ap = argparse.ArgumentParser(prog="python -m core.config", description="Eidos E3 config loader")
    ap.add_argument("--dir", default="cfg", help="config directory (default: cfg)")
    ap.add_argument("--print", action="store_true", help="pretty print summary")
    ap.add_argument("--json", action="store_true", help="dump full JSON")
    ap.add_argument("--validate", action="store_true", help="exit 0 if valid, else 1")
    args = ap.parse_args()

    try:
        cfg = load_all(args.dir)
    except Exception as e:
        if args.validate:
            print(f"[config] INVALID: {e}", file=sys.stderr); sys.exit(1)
        raise

    if args.print:
        print(f"[config] loaded from {Path(args.dir).resolve()}")
        print(f"  self.name: {cfg.self.name}")
        print(f"  workspace: {cfg.self.workspace}")
        print(f"  drives: {', '.join(sorted(cfg.drives.drives.keys()))}")
        print(f"  scopes: {', '.join(sorted(cfg.budgets.scopes.keys()))}")
        print(f"  rungs: {', '.join(r.name for r in cfg.policies.autonomy_ladder)}")
        print(f"  skills: {', '.join(sorted(cfg.skills.competencies.keys()))}")
    if args.json:
        print(json.dumps(to_dict(cfg), indent=2))
    if args.validate and not (args.print or args.json):
        print("[config] OK")

if __name__ == "__main__":  # pragma: no cover
    _cli()
