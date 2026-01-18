import importlib
import logging
import sys
import textwrap
from functools import wraps
from typing import Any, Callable, Iterable, Optional, TypeVar, Union
from packaging.version import Version
from ray._private.thirdparty.tabulate.tabulate import tabulate
from ray.util.annotations import DeveloperAPI
from ray.widgets import Template
def _has_outdated(*deps: Iterable[Union[str, Optional[str]]], message: Optional[str]=None):
    outdated = []
    for lib, version in deps:
        try:
            module = importlib.import_module(lib)
            if version and Version(module.__version__) < Version(version):
                outdated.append([lib, version, module.__version__])
        except ImportError:
            pass
    if outdated:
        outdated_strs = []
        install_args = []
        for lib, version, installed in outdated:
            outdated_strs.append(f'{lib}=={installed} found, needs {lib}>={version}')
            install_args.append(f'{lib}>={version}')
        outdated_str = textwrap.indent('\n'.join(outdated_strs), '  ')
        install_str = ' '.join(install_args)
        if not message:
            message = f'Run `pip install -U {install_str}` for rich notebook output.'
        logger.info(f'Outdated packages:\n{outdated_str}\n{message}', stacklevel=3)
    return outdated