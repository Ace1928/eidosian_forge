import copy
import logging
import os
import re
from typing import Collection, Dict, List, Optional, Set, Tuple, Union
from pip._vendor.packaging.markers import Marker
from pip._vendor.packaging.requirements import InvalidRequirement, Requirement
from pip._vendor.packaging.specifiers import Specifier
from pip._internal.exceptions import InstallationError
from pip._internal.models.index import PyPI, TestPyPI
from pip._internal.models.link import Link
from pip._internal.models.wheel import Wheel
from pip._internal.req.req_file import ParsedRequirement
from pip._internal.req.req_install import InstallRequirement
from pip._internal.utils.filetypes import is_archive_file
from pip._internal.utils.misc import is_installable_dir
from pip._internal.utils.packaging import get_requirement
from pip._internal.utils.urls import path_to_url
from pip._internal.vcs import is_url, vcs
def _get_url_from_path(path: str, name: str) -> Optional[str]:
    """
    First, it checks whether a provided path is an installable directory. If it
    is, returns the path.

    If false, check if the path is an archive file (such as a .whl).
    The function checks if the path is a file. If false, if the path has
    an @, it will treat it as a PEP 440 URL requirement and return the path.
    """
    if _looks_like_path(name) and os.path.isdir(path):
        if is_installable_dir(path):
            return path_to_url(path)
        raise InstallationError(f"Directory {name!r} is not installable. Neither 'setup.py' nor 'pyproject.toml' found.")
    if not is_archive_file(path):
        return None
    if os.path.isfile(path):
        return path_to_url(path)
    urlreq_parts = name.split('@', 1)
    if len(urlreq_parts) >= 2 and (not _looks_like_path(urlreq_parts[0])):
        return None
    logger.warning('Requirement %r looks like a filename, but the file does not exist', name)
    return path_to_url(path)