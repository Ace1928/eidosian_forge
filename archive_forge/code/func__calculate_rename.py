import difflib
from pathlib import Path
from typing import Dict, Iterable, Tuple
from parso import split_lines
from jedi.api.exceptions import RefactoringError
from jedi.inference.value.namespace import ImplicitNSName
def _calculate_rename(path, new_name):
    dir_ = path.parent
    if path.name in ('__init__.py', '__init__.pyi'):
        return (dir_, dir_.parent.joinpath(new_name))
    return (path, dir_.joinpath(new_name + path.suffix))