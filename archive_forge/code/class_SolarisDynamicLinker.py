from __future__ import annotations
import abc
import os
import typing as T
import re
from .base import ArLikeLinker, RSPFileSyntax
from .. import mesonlib
from ..mesonlib import EnvironmentException, MesonException
from ..arglist import CompilerArgs
class SolarisDynamicLinker(PosixDynamicLinkerMixin, DynamicLinker):
    """Sys-V derived linker used on Solaris and OpenSolaris."""
    id = 'ld.solaris'

    def get_link_whole_for(self, args: T.List[str]) -> T.List[str]:
        if not args:
            return args
        return self._apply_prefix('--whole-archive') + args + self._apply_prefix('--no-whole-archive')

    def get_pie_args(self) -> T.List[str]:
        pc, stdo, stde = mesonlib.Popen_safe(self.exelist + self._apply_prefix('-zhelp'))
        for line in (stdo + stde).split('\n'):
            if '-z type' in line:
                if 'pie' in line:
                    return ['-z', 'type=pie']
                break
        return []

    def get_asneeded_args(self) -> T.List[str]:
        return self._apply_prefix(['-z', 'ignore'])

    def no_undefined_args(self) -> T.List[str]:
        return ['-z', 'defs']

    def get_allow_undefined_args(self) -> T.List[str]:
        return ['-z', 'nodefs']

    def fatal_warnings(self) -> T.List[str]:
        return ['-z', 'fatal-warnings']

    def build_rpath_args(self, env: 'Environment', build_dir: str, from_dir: str, rpath_paths: T.Tuple[str, ...], build_rpath: str, install_rpath: str) -> T.Tuple[T.List[str], T.Set[bytes]]:
        if not rpath_paths and (not install_rpath) and (not build_rpath):
            return ([], set())
        processed_rpaths = prepare_rpaths(rpath_paths, build_dir, from_dir)
        all_paths = mesonlib.OrderedSet([os.path.join('$ORIGIN', p) for p in processed_rpaths])
        rpath_dirs_to_remove: T.Set[bytes] = set()
        for p in all_paths:
            rpath_dirs_to_remove.add(p.encode('utf8'))
        if build_rpath != '':
            all_paths.add(build_rpath)
            for p in build_rpath.split(':'):
                rpath_dirs_to_remove.add(p.encode('utf8'))
        paths = ':'.join(all_paths)
        if len(paths) < len(install_rpath):
            padding = 'X' * (len(install_rpath) - len(paths))
            if not paths:
                paths = padding
            else:
                paths = paths + ':' + padding
        return (self._apply_prefix(f'-rpath,{paths}'), rpath_dirs_to_remove)

    def get_soname_args(self, env: 'Environment', prefix: str, shlib_name: str, suffix: str, soversion: str, darwin_versions: T.Tuple[str, str]) -> T.List[str]:
        sostr = '' if soversion is None else '.' + soversion
        return self._apply_prefix(f'-soname,{prefix}{shlib_name}.{suffix}{sostr}')