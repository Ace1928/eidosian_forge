from __future__ import annotations
import glob
import os
import re
import pathlib
import shutil
import subprocess
import typing as T
import functools
from mesonbuild.interpreterbase.decorators import FeatureDeprecated
from .. import mesonlib, mlog
from ..environment import get_llvm_tool_names
from ..mesonlib import version_compare, version_compare_many, search_version, stringlistify, extract_as_list
from .base import DependencyException, DependencyMethods, detect_compiler, strip_system_includedirs, strip_system_libdirs, SystemDependency, ExternalDependency, DependencyTypeName
from .cmake import CMakeDependency
from .configtool import ConfigToolDependency
from .detect import packages
from .factory import DependencyFactory
from .misc import threads_factory
from .pkgconfig import PkgConfigDependency
def __fix_bogus_link_args(self, args: T.List[str]) -> T.List[str]:
    """This function attempts to fix bogus link arguments that llvm-config
        generates.

        Currently it works around the following:
            - FreeBSD: when statically linking -l/usr/lib/libexecinfo.so will
              be generated, strip the -l in cases like this.
            - Windows: We may get -LIBPATH:... which is later interpreted as
              "-L IBPATH:...", if we're using an msvc like compilers convert
              that to "/LIBPATH", otherwise to "-L ..."
        """
    new_args = []
    for arg in args:
        if arg.startswith('-l') and arg.endswith('.so'):
            new_args.append(arg.lstrip('-l'))
        elif arg.startswith('-LIBPATH:'):
            cpp = self.env.coredata.compilers[self.for_machine]['cpp']
            new_args.extend(cpp.get_linker_search_args(arg.lstrip('-LIBPATH:')))
        else:
            new_args.append(arg)
    return new_args