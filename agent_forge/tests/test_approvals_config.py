import importlib
import sys
from pathlib import Path
from actuators import approvals as A


def test_template_config_allows_and_denies(tmp_path):
    ok, _ = A.allowed_cmd(["bash", "-lc", "echo hi"], ".", template="hygiene")
    assert ok
    ok2, _ = A.allowed_cmd(["docker", "ps"], ".", template="hygiene")
    assert not ok2


def test_missing_config_fallback(tmp_path):
    cfg = Path("cfg/approvals.yaml")
    backup = tmp_path / "approvals.yaml"
    cfg.rename(backup)
    try:
        sys.modules.pop("actuators.approvals", None)
        A2 = importlib.import_module("actuators.approvals")
        ok, _ = A2.allowed_cmd(["bash"], ".", template=None)
        assert ok
    finally:
        backup.rename(cfg)
        sys.modules.pop("actuators.approvals", None)
        importlib.import_module("actuators.approvals")
