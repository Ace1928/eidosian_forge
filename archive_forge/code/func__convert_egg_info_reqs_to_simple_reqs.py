from __future__ import annotations
import os
import re
import abc
import csv
import sys
import json
import zipp
import email
import types
import inspect
import pathlib
import operator
import textwrap
import warnings
import functools
import itertools
import posixpath
import collections
from . import _adapters, _meta, _py39compat
from ._collections import FreezableDefaultDict, Pair
from ._compat import (
from ._functools import method_cache, pass_none
from ._itertools import always_iterable, unique_everseen
from ._meta import PackageMetadata, SimplePath
from contextlib import suppress
from importlib import import_module
from importlib.abc import MetaPathFinder
from itertools import starmap
from typing import Any, Iterable, List, Mapping, Match, Optional, Set, cast
@staticmethod
def _convert_egg_info_reqs_to_simple_reqs(sections):
    """
        Historically, setuptools would solicit and store 'extra'
        requirements, including those with environment markers,
        in separate sections. More modern tools expect each
        dependency to be defined separately, with any relevant
        extras and environment markers attached directly to that
        requirement. This method converts the former to the
        latter. See _test_deps_from_requires_text for an example.
        """

    def make_condition(name):
        return name and f'extra == "{name}"'

    def quoted_marker(section):
        section = section or ''
        extra, sep, markers = section.partition(':')
        if extra and markers:
            markers = f'({markers})'
        conditions = list(filter(None, [markers, make_condition(extra)]))
        return '; ' + ' and '.join(conditions) if conditions else ''

    def url_req_space(req):
        """
            PEP 508 requires a space between the url_spec and the quoted_marker.
            Ref python/importlib_metadata#357.
            """
        return ' ' * ('@' in req)
    for section in sections:
        space = url_req_space(section.value)
        yield (section.value + space + quoted_marker(section.name))