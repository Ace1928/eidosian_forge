from __future__ import annotations
import os
import re
import subprocess
import typing as T
from .. import mlog
from .. import mesonlib
from ..compilers.compilers import CrossNoRunException
from ..mesonlib import (
from ..environment import detect_cpu_family
from .base import DependencyException, DependencyMethods, DependencyTypeName, SystemDependency
from .configtool import ConfigToolDependency
from .detect import packages
from .factory import DependencyFactory
class SDL2DependencyConfigTool(ConfigToolDependency):
    tools = ['sdl2-config']
    tool_name = 'sdl2-config'

    def __init__(self, name: str, environment: 'Environment', kwargs: T.Dict[str, T.Any]):
        super().__init__(name, environment, kwargs)
        if not self.is_found:
            return
        self.compile_args = self.get_config_value(['--cflags'], 'compile_args')
        self.link_args = self.get_config_value(['--libs'], 'link_args')