from __future__ import annotations
import logging
import os.path
from typing import Callable
from typing import Generator
from typing import Sequence
from flake8 import utils
def _filenames_from(arg: str, *, predicate: Callable[[str], bool]) -> Generator[str, None, None]:
    """Generate filenames from an argument.

    :param arg:
        Parameter from the command-line.
    :param predicate:
        Predicate to use to filter out filenames. If the predicate
        returns ``True`` we will exclude the filename, otherwise we
        will yield it. By default, we include every filename
        generated.
    :returns:
        Generator of paths
    """
    if predicate(arg):
        return
    if os.path.isdir(arg):
        for root, sub_directories, files in os.walk(arg):
            for directory in tuple(sub_directories):
                joined = os.path.join(root, directory)
                if predicate(joined):
                    sub_directories.remove(directory)
            for filename in files:
                joined = os.path.join(root, filename)
                if not predicate(joined):
                    yield joined
    else:
        yield arg