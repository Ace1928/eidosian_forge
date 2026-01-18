import glob
import importlib
import inspect
import logging
import os
import re
import sys
from typing import Iterable, List, Optional, Tuple, Union
def _update_link_based_imported_package(link: str, pkg_ver: str, version_digits: Optional[int]) -> str:
    """Adjust the linked external docs to be local.

    Args:
        link: the source link to be replaced
        pkg_ver: the target link to be replaced, if ``{package.version}`` is included it will be replaced accordingly
        version_digits: for semantic versioning, how many digits to be considered

    """
    pkg_att = pkg_ver.split('.')
    try:
        ver = _load_pypi_versions(pkg_att[0])[-1]
    except Exception:
        module = importlib.import_module('.'.join(pkg_att[:-1]))
        ver = getattr(module, pkg_att[0])
    ver = ver.split('+')[0]
    ver = '.'.join(ver.split('.')[:version_digits])
    return link.replace(f'{{{pkg_ver}}}', ver)