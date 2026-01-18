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
def _get_module_mapping(self) -> Iterator[Tuple[str, str]]:
    """Iterate over all modules producing (dest, src) pairs."""
    for package, module, module_file in self.find_all_modules():
        package = package.split('.')
        filename = self.get_module_outfile(self.build_lib, package, module)
        yield (filename, module_file)