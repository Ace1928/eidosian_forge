from __future__ import annotations
from .base import ExternalDependency, DependencyException, DependencyTypeName
from .pkgconfig import PkgConfigDependency
from ..mesonlib import (Popen_safe, OptionKey, join_args, version_compare)
from ..programs import ExternalProgram
from .. import mlog
import re
import os
import json
import typing as T
def _check_dub(self) -> T.Optional[T.Tuple[ExternalProgram, str]]:

    def find() -> T.Optional[T.Tuple[ExternalProgram, str]]:
        dubbin = ExternalProgram('dub', silent=True)
        if not dubbin.found():
            return None
        try:
            p, out = Popen_safe(dubbin.get_command() + ['--version'])[0:2]
            if p.returncode != 0:
                mlog.warning("Found dub {!r} but couldn't run it".format(' '.join(dubbin.get_command())))
                return None
        except (FileNotFoundError, PermissionError):
            return None
        vermatch = re.search('DUB version (\\d+\\.\\d+\\.\\d+.*), ', out.strip())
        if vermatch:
            dubver = vermatch.group(1)
        else:
            mlog.warning(f"Found dub {' '.join(dubbin.get_command())} but couldn't parse version in {out.strip()}")
            return None
        return (dubbin, dubver)
    found = find()
    if found is None:
        mlog.log('Found DUB:', mlog.red('NO'))
    else:
        dubbin, dubver = found
        mlog.log('Found DUB:', mlog.bold(dubbin.get_path()), '(version %s)' % dubver)
    return found