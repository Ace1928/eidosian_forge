from functools import partial
from glob import glob
from distutils.util import convert_path
import distutils.command.build_py as orig
import os
import fnmatch
import textwrap
import io
import distutils.errors
import itertools
import stat
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple
from ..extern.more_itertools import unique_everseen
from ..warnings import SetuptoolsDeprecationWarning
def _filter_build_files(self, files: Iterable[str], egg_info: str) -> Iterator[str]:
    """
        ``build_meta`` may try to create egg_info outside of the project directory,
        and this can be problematic for certain plugins (reported in issue #3500).

        Extensions might also include between their sources files created on the
        ``build_lib`` and ``build_temp`` directories.

        This function should filter this case of invalid files out.
        """
    build = self.get_finalized_command('build')
    build_dirs = (egg_info, self.build_lib, build.build_temp, build.build_base)
    norm_dirs = [os.path.normpath(p) for p in build_dirs if p]
    for file in files:
        norm_path = os.path.normpath(file)
        if not os.path.isabs(file) or all((d not in norm_path for d in norm_dirs)):
            yield file