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
def _build_one(req: InstallRequirement, output_dir: str, verify: bool, build_options: List[str], global_options: List[str], editable: bool) -> Optional[str]:
    """Build one wheel.

    :return: The filename of the built wheel, or None if the build failed.
    """
    artifact = 'editable' if editable else 'wheel'
    try:
        ensure_dir(output_dir)
    except OSError as e:
        logger.warning('Building %s for %s failed: %s', artifact, req.name, e)
        return None
    with req.build_env:
        wheel_path = _build_one_inside_env(req, output_dir, build_options, global_options, editable)
    if wheel_path and verify:
        try:
            _verify_one(req, wheel_path)
        except (InvalidWheelFilename, UnsupportedWheel) as e:
            logger.warning('Built %s for %s is invalid: %s', artifact, req.name, e)
            return None
    return wheel_path