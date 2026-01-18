from __future__ import annotations
import collections
import itertools
import json
import logging
import os
import platform
import sys
import warnings
from typing import TYPE_CHECKING
import requests
from monty.json import MontyDecoder
from pymatgen.core import SETTINGS
from pymatgen.core import __version__ as PMG_VERSION
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
class MPRester:
    """A class to conveniently interface with the new and legacy Materials Project REST interface.

    The recommended way to use MPRester is as a context manager to ensure
    that sessions are properly closed after usage:

        with MPRester("API_KEY") as mpr:
            docs = mpr.call_some_method()

    MPRester uses the "requests" package, which provides HTTP connection
    pooling. All connections are made via https for security.

    For more advanced uses of the Materials API, please consult the API
    documentation at https://materialsproject.org/api and https://docs.materialsproject.org.

    This class handles the transition between old and new MP API, making it easy to switch between them
    by passing a new (length 32) or old (15 <= length <= 17) API key. See https://docs.materialsproject.org
    for which API to use.
    """

    def __new__(cls, *args, **kwargs) -> _MPResterNew | _MPResterBasic | _MPResterLegacy:
        """
        Args:
           *args: Pass through to either legacy or new MPRester.
           **kwargs: Pass through to either legacy or new MPRester.
        """
        api_key = args[0] if len(args) > 0 else None
        if api_key is None:
            api_key = kwargs.get('api_key', SETTINGS.get('PMG_MAPI_KEY'))
            kwargs['api_key'] = api_key
        if not api_key:
            raise ValueError('Please supply an API key. See https://materialsproject.org/api for details.')
        if len(api_key) != 32:
            from pymatgen.ext.matproj_legacy import _MPResterLegacy
            return _MPResterLegacy(*args, **kwargs)
        try:
            from mp_api.client import MPRester as _MPResterNew
            return _MPResterNew(*args, **kwargs)
        except Exception:
            return _MPResterBasic(*args, **kwargs)