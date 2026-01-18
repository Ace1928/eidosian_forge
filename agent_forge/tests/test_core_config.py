import os
import json
import shutil
from pathlib import Path

from core import config as C

def make_tmp_cfg(tmp_path: Path):
    # copy from sample to avoid writing fixtures manually
    src = Path("cfg.sample")
    dst = tmp_path / "cfg"
    shutil.copytree(src, dst)
    return dst

def test_load_and_print(tmp_path, monkeypatch):
    cfg_dir = make_tmp_cfg(tmp_path)
    monkeypatch.setenv("EIDOS_WS", "/tmp/eidos")
    # add an env var and check expansion by editing workspace on the fly
    p = cfg_dir / "self.yaml"
    text = p.read_text(encoding="utf-8").replace("workspace: .", "workspace: ${EIDOS_WS}")
    p.write_text(text, encoding="utf-8")

    cfg = C.load_all(cfg_dir)
    assert cfg.self.name == "Eidos"
    assert cfg.self.workspace == "/tmp/eidos"
    d = C.to_dict(cfg)
    j = json.dumps(d)  # must be serializable


def test_validation_error_on_type(tmp_path):
    cfg_dir = make_tmp_cfg(tmp_path)
    p = cfg_dir / "drives.yaml"
    p.write_text(p.read_text().replace("target: 7", 'target: "seven"'), encoding="utf-8")
    try:
        C.load_all(cfg_dir)
        assert False, "should have failed"
    except (TypeError, ValueError) as e:
        assert "drives.yaml" in str(e)
