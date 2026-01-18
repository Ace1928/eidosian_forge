import os
from typing import Callable, Generator, Union
def _exclude_fn(path: str, root: str) -> bool:
    return exclude_fn(path, root) if len(inspect.signature(exclude_fn).parameters) == 2 else exclude_fn(path)