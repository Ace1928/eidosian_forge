from __future__ import annotations
import subprocess, os, signal, resource, tempfile
from pathlib import Path
from typing import Sequence

class SandboxError(RuntimeError): ...

def _limits(cpu_s: float, mem_mb: int):
    def preexec():
        # hard cap CPU & address space; lower file descriptors
        resource.setrlimit(resource.RLIMIT_CPU, (int(cpu_s), int(cpu_s)))
        mem = int(mem_mb) * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
        os.setsid()  # group for termination
    return preexec

def run_sandboxed(cmd: Sequence[str], *, cwd: str, timeout_s: float,
                  cpu_quota_s: float = 60, mem_mb: int = 1024) -> tuple[int, bytes, bytes]:
    """Execute safely with rlimits. Returns (rc, stdout, stderr)."""
    cwdp = Path(cwd)
    cwdp.mkdir(parents=True, exist_ok=True)
    try:
        p = subprocess.Popen(
            list(map(str, cmd)),
            cwd=str(cwdp),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=_limits(cpu_quota_s, mem_mb),
            text=False,
            env={**os.environ, "LC_ALL":"C.UTF-8"}
        )
        # allow a small grace period beyond the requested timeout to
        # accommodate process start-up overhead in constrained environments
        out, err = p.communicate(timeout=timeout_s + 1)
        return p.returncode, out or b"", err or b""
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except Exception:
            pass
        raise SandboxError("timeout")
