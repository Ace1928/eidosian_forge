from distutils import log
import distutils.command.sdist as orig
import os
import sys
import io
import contextlib
from itertools import chain
from .._importlib import metadata
from .build import _ORIGINAL_SUBCOMMANDS
def _add_defaults_build_sub_commands(self):
    build = self.get_finalized_command('build')
    missing_cmds = set(build.get_sub_commands()) - _ORIGINAL_SUBCOMMANDS
    cmds = (self.get_finalized_command(c) for c in missing_cmds)
    files = (c.get_source_files() for c in cmds if hasattr(c, 'get_source_files'))
    self.filelist.extend(chain.from_iterable(files))