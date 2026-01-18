import contextlib
import os
import re
import sys
from distutils.core import Command
from distutils.errors import *
from distutils.sysconfig import customize_compiler, get_python_version
from distutils.sysconfig import get_config_h_filename
from distutils.dep_util import newer_group
from distutils.extension import Extension
from distutils.util import get_platform
from distutils import log
from site import USER_BASE
def _build_extensions_parallel(self):
    workers = self.parallel
    if self.parallel is True:
        workers = os.cpu_count()
    try:
        from concurrent.futures import ThreadPoolExecutor
    except ImportError:
        workers = None
    if workers is None:
        self._build_extensions_serial()
        return
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(self.build_extension, ext) for ext in self.extensions]
        for ext, fut in zip(self.extensions, futures):
            with self._filter_build_errors(ext):
                fut.result()