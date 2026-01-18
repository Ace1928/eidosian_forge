from __future__ import annotations
import json
import pickle  # use pickle, not cPickle so that we get the traceback in case of errors
import string
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar
from unittest import TestCase
import pytest
from monty.json import MontyDecoder, MontyEncoder, MSONable
from monty.serialization import loadfn
from pymatgen.core import ROOT, SETTINGS, Structure
@pytest.fixture(autouse=True)
def _tmp_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    self.tmp_path = tmp_path