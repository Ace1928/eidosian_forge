import functools
import os
import re
import _distutils_hack.override  # noqa: F401
import distutils.core
from distutils.errors import DistutilsOptionError
from distutils.util import convert_path as _convert_path
from . import logging, monkey
from . import version as _version_module
from .depends import Require
from .discovery import PackageFinder, PEP420PackageFinder
from .dist import Distribution
from .extension import Extension
from .warnings import SetuptoolsDeprecationWarning
class MinimalDistribution(distutils.core.Distribution):
    """
        A minimal version of a distribution for supporting the
        fetch_build_eggs interface.
        """

    def __init__(self, attrs):
        _incl = ('dependency_links', 'setup_requires')
        filtered = {k: attrs[k] for k in set(_incl) & set(attrs)}
        super().__init__(filtered)
        self.set_defaults._disable()

    def _get_project_config_files(self, filenames=None):
        """Ignore ``pyproject.toml``, they are not related to setup_requires"""
        try:
            cfg, toml = super()._split_standard_project_metadata(filenames)
            return (cfg, ())
        except Exception:
            return (filenames, ())

    def finalize_options(self):
        """
            Disable finalize_options to avoid building the working set.
            Ref #2158.
            """