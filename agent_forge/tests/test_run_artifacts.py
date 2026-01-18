from pathlib import Path
import subprocess
from core.artifacts import write_run_artifacts, read_run_artifacts
from actuators.shell_exec import run_step


def test_write_and_read_artifacts(tmp_path: Path):
    base = tmp_path / "state"
    run_id = "r1"
    write_run_artifacts(base, run_id, b"hello\n", b"warn\n", {"k": 1})
    data = read_run_artifacts(base, run_id)
    assert data["stdout"].strip() == b"hello"
    assert data["stderr"].strip() == b"warn"
    assert data["meta"]["k"] == 1


def test_runner_writes_artifacts_and_cli(tmp_path: Path):
    base = tmp_path / "state"
    base.mkdir(parents=True, exist_ok=True)
    res = run_step(str(base), "step-1",
                   ["bash", "-lc", "echo hi; echo oops 1>&2; exit 1"],
                   cwd=".", budget_s=5.0)
    assert res["status"] in ("ok", "timeout", "error", "denied") or res["status"] == "timeout"
    run_id = res.get("run_id")
    assert run_id, "runner should return a run_id"
    d = base / "runs" / run_id
    assert (d / "stdout.txt").exists()
    assert (d / "stderr.txt").exists()

    # CLI interaction
    cmd_ls = ["python", "bin/eidctl", "runs", "ls", "--dir", str(base)]
    out = subprocess.check_output(cmd_ls, text=True)
    assert run_id in out

    cmd_show = ["python", "bin/eidctl", "runs", "show", "--dir", str(base), "--run", run_id, "--head", "10"]
    out_show = subprocess.check_output(cmd_show, text=True)
    assert "hi" in out_show
