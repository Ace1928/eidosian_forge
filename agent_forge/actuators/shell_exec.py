from __future__ import annotations
from datetime import datetime, timezone
from typing import Sequence
from core.state import add_run  # DB write
from .approvals import allowed_cmd
from .sandbox import run_sandboxed, SandboxError
from core.artifacts import write_run_artifacts

def run_step(base_dir: str, step_id: str, cmd: Sequence[str], *, cwd: str,
             budget_s: float, template: str | None = None) -> dict:
    """Gate + execute + persist run row. Returns run dict (for logs)."""
    ok, reason = allowed_cmd(cmd, cwd, template=template)
    if not ok:
        return {"status":"denied","reason":reason}

    start = _now()
    try:
        rc, out, err = run_sandboxed(cmd, cwd=cwd, timeout_s=budget_s,
                                     cpu_quota_s=budget_s, mem_mb=1024)
        end = _now()
        r = add_run(base_dir, step_id, start, end, rc, len(out) + len(err), "")
        try:
            write_run_artifacts(base_dir, r.id, out, err,
                                {"cmd": list(cmd), "cwd": str(cwd), "rc": int(rc), "status": "ok"})
        except Exception:
            pass
        return {"status":"ok","rc":rc,"run_id":r.id,"bytes":r.bytes_out}
    except SandboxError as e:
        end = _now()
        r = add_run(base_dir, step_id, start, end, None, 0, str(e))
        try:
            write_run_artifacts(base_dir, r.id, b"", b"",
                                {"cmd": list(cmd), "cwd": str(cwd), "status":"timeout"})
        except Exception:
            pass
        return {"status":"timeout","run_id":r.id}
    except Exception as e:
        end = _now()
        r = add_run(base_dir, step_id, start, end, None, 0, f"error:{e}")
        try:
            write_run_artifacts(base_dir, r.id, b"", b"",
                                {"cmd": list(cmd), "cwd": str(cwd), "status":"error", "error": str(e)})
        except Exception:
            pass
        return {"status":"error","run_id":r.id}

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
