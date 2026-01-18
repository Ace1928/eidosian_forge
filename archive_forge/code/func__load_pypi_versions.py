import glob
import importlib
import inspect
import logging
import os
import re
import sys
from typing import Iterable, List, Optional, Tuple, Union
def _load_pypi_versions(package_name: str) -> List[str]:
    """Load the versions of the package from PyPI.

    >>> _load_pypi_versions("numpy")  # doctest: +ELLIPSIS
    ['0.9.6', '0.9.8', '1.0', ...]
    >>> _load_pypi_versions("scikit-learn")  # doctest: +ELLIPSIS
    ['0.9', '0.10', '0.11', '0.12', ...]

    """
    from distutils.version import LooseVersion
    import requests
    url = f'https://pypi.org/pypi/{package_name}/json'
    data = requests.get(url, timeout=10).json()
    versions = data['releases'].keys()
    versions = {k for k in versions if re.match('^\\d+(\\.\\d+)*$', k)}
    return sorted(versions, key=LooseVersion)