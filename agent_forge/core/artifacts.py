from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, Literal

__all__ = ["run_dir", "write_blob", "write_run_artifacts", "read_run_artifacts"]


def run_dir(base: str | Path, run_id: str) -> Path:
    """Return directory path for a run under ``base`` and ensure it exists."""
    p = Path(base) / "runs" / str(run_id)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _safe_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
    tmp.replace(path)  # atomic on POSIX


def write_blob(base: str | Path, run_id: str, name: Literal["stdout", "stderr"], data: bytes) -> Path:
    """Persist a single blob under the run directory and return its path."""
    path = run_dir(base, run_id) / f"{name}.txt"
    _safe_write(path, data or b"")
    return path


def write_run_artifacts(base: str | Path, run_id: str,
                        stdout_bytes: bytes, stderr_bytes: bytes,
                        meta: Dict[str, Any]) -> Path:
    """
    Persist run artifacts under ``state/runs/<run_id>/``:
      - stdout.txt
      - stderr.txt
      - meta.json
    Returns the directory path created.
    """
    base_dir = run_dir(base, run_id)
    write_blob(base, run_id, "stdout", stdout_bytes)
    write_blob(base, run_id, "stderr", stderr_bytes)
    _safe_write(base_dir / "meta.json", json.dumps(meta or {}, ensure_ascii=False, indent=2).encode("utf-8"))
    return base_dir


def read_run_artifacts(base: str | Path, run_id: str) -> Dict[str, Any]:
    """Load artifacts; missing files are treated as empty/defaults."""
    base_dir = run_dir(base, run_id)
    stdout_p = base_dir / "stdout.txt"
    stderr_p = base_dir / "stderr.txt"
    meta_p = base_dir / "meta.json"
    out = stdout_p.read_bytes() if stdout_p.exists() else b""
    err = stderr_p.read_bytes() if stderr_p.exists() else b""
    if meta_p.exists():
        try:
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    else:
        meta = {}
    return {"stdout": out, "stderr": err, "meta": meta}

