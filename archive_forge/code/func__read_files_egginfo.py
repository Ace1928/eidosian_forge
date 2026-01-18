import os
import re
import abc
import csv
import sys
import email
import pathlib
import zipfile
import operator
import textwrap
import warnings
import functools
import itertools
import posixpath
import collections
from . import _adapters, _meta
from ._collections import FreezableDefaultDict, Pair
from ._functools import method_cache, pass_none
from ._itertools import always_iterable, unique_everseen
from ._meta import PackageMetadata, SimplePath
from contextlib import suppress
from importlib import import_module
from importlib.abc import MetaPathFinder
from itertools import starmap
from typing import List, Mapping, Optional, Union
def _read_files_egginfo(self):
    """
        SOURCES.txt might contain literal commas, so wrap each line
        in quotes.
        """
    text = self.read_text('SOURCES.txt')
    return text and map('"{}"'.format, text.splitlines())