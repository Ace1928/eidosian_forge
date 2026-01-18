import logging
import os
from collections.abc import Mapping
from email.headerregistry import Address
from functools import partial, reduce
from inspect import cleandoc
from itertools import chain
from types import MappingProxyType
from typing import (
from ..errors import RemovedConfigError
from ..warnings import SetuptoolsWarning
def _copy_command_options(pyproject: dict, dist: 'Distribution', filename: _Path):
    tool_table = pyproject.get('tool', {})
    cmdclass = tool_table.get('setuptools', {}).get('cmdclass', {})
    valid_options = _valid_command_options(cmdclass)
    cmd_opts = dist.command_options
    for cmd, config in pyproject.get('tool', {}).get('distutils', {}).items():
        cmd = json_compatible_key(cmd)
        valid = valid_options.get(cmd, set())
        cmd_opts.setdefault(cmd, {})
        for key, value in config.items():
            key = json_compatible_key(key)
            cmd_opts[cmd][key] = (str(filename), value)
            if key not in valid:
                _logger.warning(f'Command option {cmd}.{key} is not defined')