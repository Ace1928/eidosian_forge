import os, time
from actuators.sandbox import run_sandboxed, SandboxError

def test_timeout(tmp_path):
    try:
        run_sandboxed(["bash","-lc","sleep 2"], cwd=str(tmp_path), timeout_s=0.3)
        assert False, "should timeout"
    except SandboxError:
        pass

def test_stdout_stderr(tmp_path):
    rc, out, err = run_sandboxed(["bash","-lc","echo hi; echo oops 1>&2; exit 3"],
                                 cwd=str(tmp_path), timeout_s=2)
    assert rc == 3 and out.strip()==b"hi" and b"oops" in err
