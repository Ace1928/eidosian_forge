from __future__ import annotations
import functools
import re
import typing as T
from .. import mesonlib
from .. import mlog
from .base import DependencyException, DependencyMethods
from .base import BuiltinDependency, SystemDependency
from .cmake import CMakeDependency, CMakeDependencyFactory
from .configtool import ConfigToolDependency
from .detect import packages
from .factory import DependencyFactory, factory_methods
from .pkgconfig import PkgConfigDependency
class BlocksDependency(SystemDependency):

    def __init__(self, environment: 'Environment', kwargs: T.Dict[str, T.Any]) -> None:
        super().__init__('blocks', environment, kwargs)
        self.name = 'blocks'
        self.is_found = False
        if self.env.machines[self.for_machine].is_darwin():
            self.compile_args = []
            self.link_args = []
        else:
            self.compile_args = ['-fblocks']
            self.link_args = ['-lBlocksRuntime']
            if not self.clib_compiler.has_header('Block.h', '', environment, disable_cache=True) or not self.clib_compiler.find_library('BlocksRuntime', environment, []):
                mlog.log(mlog.red('ERROR:'), 'BlocksRuntime not found.')
                return
        source = '\n            int main(int argc, char **argv)\n            {\n                int (^callback)(void) = ^ int (void) { return 0; };\n                return callback();\n            }'
        with self.clib_compiler.compile(source, extra_args=self.compile_args + self.link_args) as p:
            if p.returncode != 0:
                mlog.log(mlog.red('ERROR:'), 'Compiler does not support blocks extension.')
                return
            self.is_found = True