import glob
import os
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Any, Dict
from pkg_resources import parse_requirements
from setuptools import find_packages
def _load_py_module(name: str, location: str) -> ModuleType:
    spec = spec_from_file_location(name, location)
    assert spec, f'Failed to load module {name} from {location}'
    py = module_from_spec(spec)
    assert spec.loader, f'ModuleSpec.loader is None for {name} from {location}'
    spec.loader.exec_module(py)
    return py