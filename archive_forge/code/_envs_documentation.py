import functools
import importlib.metadata
import logging
import os
import pathlib
import sys
import zipfile
import zipimport
from typing import Iterator, List, Optional, Sequence, Set, Tuple
from pip._vendor.packaging.utils import NormalizedName, canonicalize_name
from pip._internal.metadata.base import BaseDistribution, BaseEnvironment
from pip._internal.models.wheel import Wheel
from pip._internal.utils.deprecation import deprecated
from pip._internal.utils.filetypes import WHEEL_EXTENSION
from ._compat import BadMetadata, BasePath, get_dist_name, get_info_location
from ._dists import Distribution
Find eggs in a location.

        This actually uses the old *pkg_resources* backend. We likely want to
        deprecate this so we can eventually remove the *pkg_resources*
        dependency entirely. Before that, this should first emit a deprecation
        warning for some versions when using the fallback since importing
        *pkg_resources* is slow for those who don't need it.
        