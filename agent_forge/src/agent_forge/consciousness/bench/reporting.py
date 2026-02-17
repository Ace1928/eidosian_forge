from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


def spec_hash(spec: Mapping[str, Any]) -> str:
    return hashlib.sha1(canonical_json(spec).encode("utf-8", "replace")).hexdigest()


def bench_report_root(state_dir: Path) -> Path:
    default = state_dir / "consciousness" / "bench"
    override = os.environ.get("EIDOS_CONSCIOUSNESS_BENCH_DIR")
    path = Path(override).resolve() if override else default.resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def trial_output_dir(state_dir: Path, *, name: str, trial_hash: str) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)[:48]
    out = bench_report_root(state_dir) / f"{ts}_{safe_name}_{trial_hash[:10]}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [json.dumps(dict(row), default=str) for row in rows]
    content = "\n".join(lines)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def write_summary(path: Path, lines: Sequence[str]) -> None:
    text = "\n".join(str(line) for line in lines).rstrip() + "\n"
    path.write_text(text, encoding="utf-8")


def git_revision(path: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    sha = str(proc.stdout or "").strip()
    if len(sha) < 7:
        return None
    return sha
