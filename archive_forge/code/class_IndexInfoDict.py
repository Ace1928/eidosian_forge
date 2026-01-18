from __future__ import annotations
import collections
import datetime
import functools
import json
import os
import re
import shutil
import string
from dataclasses import dataclass
from typing import Any, Iterable, TYPE_CHECKING, cast
import coverage
from coverage.data import CoverageData, add_data_to_hash
from coverage.exceptions import NoDataError
from coverage.files import flat_rootname
from coverage.misc import ensure_dir, file_be_gone, Hasher, isolate_module, format_local_datetime
from coverage.misc import human_sorted, plural, stdout_link
from coverage.report_core import get_analysis_to_report
from coverage.results import Analysis, Numbers
from coverage.templite import Templite
from coverage.types import TLineNo, TMorf
from coverage.version import __url__
class IndexInfoDict(TypedDict):
    """Information for each file, to render the index file."""
    nums: Numbers
    html_filename: str
    relative_filename: str