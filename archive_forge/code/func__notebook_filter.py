import asyncio
import os
from typing import Awaitable, Tuple, Type, TypeVar, Union
from typing import Dict as TypeDict
from typing import List as TypeList
from pathlib import Path
from traitlets.traitlets import Dict, Float, List, default
from nbclient.util import ensure_async
import re
from .notebook_renderer import NotebookRenderer
from .utils import ENV_VARIABLE
def _notebook_filter(self, nb_path: Path) -> bool:
    """Helper to filter blacklisted notebooks.

                Args:
                    nb_path (Path): Path to notebook

                Returns:
                    bool: return `False` if notebook is in `ipynb_checkpoints` folder or
                    is blacklisted, `True` otherwise.
                """
    nb_name = str(nb_path)
    if '.ipynb_checkpoints' in nb_name:
        return False
    for nb_pattern in self.preheat_blacklist:
        pattern = re.compile(nb_pattern)
        if nb_pattern in nb_name or bool(pattern.match(nb_name)):
            return False
    return True