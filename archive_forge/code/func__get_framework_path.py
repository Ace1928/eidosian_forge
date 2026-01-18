from __future__ import annotations
from .base import DependencyTypeName, ExternalDependency, DependencyException
from ..mesonlib import MesonException, Version, stringlistify
from .. import mlog
from pathlib import Path
import typing as T
def _get_framework_path(self, path: str, name: str) -> T.Optional[Path]:
    p = Path(path)
    lname = name.lower()
    for d in p.glob('*.framework/'):
        if lname == d.name.rsplit('.', 1)[0].lower():
            return d
    return None