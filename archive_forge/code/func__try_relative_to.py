import difflib
from pathlib import Path
from typing import Dict, Iterable, Tuple
from parso import split_lines
from jedi.api.exceptions import RefactoringError
from jedi.inference.value.namespace import ImplicitNSName
def _try_relative_to(path: Path, base: Path) -> Path:
    try:
        return path.relative_to(base)
    except ValueError:
        return path