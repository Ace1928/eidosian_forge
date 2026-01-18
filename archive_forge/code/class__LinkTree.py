import logging
import io
import os
import shutil
import sys
import traceback
from contextlib import suppress
from enum import Enum
from inspect import cleandoc
from itertools import chain, starmap
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import (
from .. import (
from ..discovery import find_package_path
from ..dist import Distribution
from ..warnings import (
from .build_py import build_py as build_py_cls
import sys
from importlib.machinery import ModuleSpec, PathFinder
from importlib.machinery import all_suffixes as module_suffixes
from importlib.util import spec_from_file_location
from itertools import chain
from pathlib import Path
class _LinkTree(_StaticPth):
    """
    Creates a ``.pth`` file that points to a link tree in the ``auxiliary_dir``.

    This strategy will only link files (not dirs), so it can be implemented in
    any OS, even if that means using hardlinks instead of symlinks.

    By collocating ``auxiliary_dir`` and the original source code, limitations
    with hardlinks should be avoided.
    """

    def __init__(self, dist: Distribution, name: str, auxiliary_dir: _Path, build_lib: _Path):
        self.auxiliary_dir = Path(auxiliary_dir)
        self.build_lib = Path(build_lib).resolve()
        self._file = dist.get_command_obj('build_py').copy_file
        super().__init__(dist, name, [self.auxiliary_dir])

    def __call__(self, wheel: 'WheelFile', files: List[str], mapping: Dict[str, str]):
        self._create_links(files, mapping)
        super().__call__(wheel, files, mapping)

    def _normalize_output(self, file: str) -> Optional[str]:
        with suppress(ValueError):
            path = Path(file).resolve().relative_to(self.build_lib)
            return str(path).replace(os.sep, '/')
        return None

    def _create_file(self, relative_output: str, src_file: str, link=None):
        dest = self.auxiliary_dir / relative_output
        if not dest.parent.is_dir():
            dest.parent.mkdir(parents=True)
        self._file(src_file, dest, link=link)

    def _create_links(self, outputs, output_mapping):
        self.auxiliary_dir.mkdir(parents=True, exist_ok=True)
        link_type = 'sym' if _can_symlink_files(self.auxiliary_dir) else 'hard'
        mappings = {self._normalize_output(k): v for k, v in output_mapping.items()}
        mappings.pop(None, None)
        for output in outputs:
            relative = self._normalize_output(output)
            if relative and relative not in mappings:
                self._create_file(relative, output)
        for relative, src in mappings.items():
            self._create_file(relative, src, link=link_type)

    def __enter__(self):
        msg = 'Strict editable install will be performed using a link tree.\n'
        _logger.warning(msg + _STRICT_WARNING)
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        msg = f'\n\n        Strict editable installation performed using the auxiliary directory:\n            {self.auxiliary_dir}\n\n        Please be careful to not remove this directory, otherwise you might not be able\n        to import/use your package.\n        '
        InformationOnly.emit('Editable installation.', msg)