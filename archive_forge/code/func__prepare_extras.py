import glob
import os
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Any, Dict
from pkg_resources import parse_requirements
from setuptools import find_packages
def _prepare_extras() -> Dict[str, Any]:
    assistant = _load_assistant()
    common_args = {'path_dir': _PATH_REQUIREMENTS, 'unfreeze': 'none' if _FREEZE_REQUIREMENTS else 'all'}
    req_files = [Path(p) for p in glob.glob(os.path.join(_PATH_REQUIREMENTS, '*.txt'))]
    extras = {p.stem: assistant.load_requirements(file_name=p.name, **common_args) for p in req_files if p.name not in ('docs.txt', 'base.txt')}
    for req in parse_requirements(extras['strategies']):
        extras[req.key] = [str(req)]
    extras['all'] = extras['strategies'] + extras['examples']
    extras['dev'] = extras['all'] + extras['test']
    return extras