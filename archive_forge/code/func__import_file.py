import sys
from importlib import import_module
from os import PathLike
from pathlib import Path
from types import ModuleType
from typing import Union
from scrapy.commands import BaseRunSpiderCommand
from scrapy.exceptions import UsageError
from scrapy.utils.spider import iter_spider_classes
def _import_file(filepath: Union[str, PathLike]) -> ModuleType:
    abspath = Path(filepath).resolve()
    if abspath.suffix not in ('.py', '.pyw'):
        raise ValueError(f'Not a Python source file: {abspath}')
    dirname = str(abspath.parent)
    sys.path = [dirname] + sys.path
    try:
        module = import_module(abspath.stem)
    finally:
        sys.path.pop(0)
    return module