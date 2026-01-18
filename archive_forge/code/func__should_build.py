import logging
import os.path
import re
import shutil
from typing import Iterable, List, Optional, Tuple
from pip._vendor.packaging.utils import canonicalize_name, canonicalize_version
from pip._vendor.packaging.version import InvalidVersion, Version
from pip._internal.cache import WheelCache
from pip._internal.exceptions import InvalidWheelFilename, UnsupportedWheel
from pip._internal.metadata import FilesystemWheel, get_wheel_distribution
from pip._internal.models.link import Link
from pip._internal.models.wheel import Wheel
from pip._internal.operations.build.wheel import build_wheel_pep517
from pip._internal.operations.build.wheel_editable import build_wheel_editable
from pip._internal.operations.build.wheel_legacy import build_wheel_legacy
from pip._internal.req.req_install import InstallRequirement
from pip._internal.utils.logging import indent_log
from pip._internal.utils.misc import ensure_dir, hash_file
from pip._internal.utils.setuptools_build import make_setuptools_clean_args
from pip._internal.utils.subprocess import call_subprocess
from pip._internal.utils.temp_dir import TempDirectory
from pip._internal.utils.urls import path_to_url
from pip._internal.vcs import vcs
def _should_build(req: InstallRequirement, need_wheel: bool) -> bool:
    """Return whether an InstallRequirement should be built into a wheel."""
    if req.constraint:
        return False
    if req.is_wheel:
        if need_wheel:
            logger.info('Skipping %s, due to already being wheel.', req.name)
        return False
    if need_wheel:
        return True
    if not req.source_dir:
        return False
    if req.editable:
        return req.supports_pyproject_editable()
    return True