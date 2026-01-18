import itertools
import re
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Union
import srsly
from wasabi import Printer
from ..tokens import Doc, DocBin
from ..training import docs_to_json
from ..training.converters import (
from ._util import Arg, Opt, app, walk_directory
class FileTypes(str, Enum):
    json = 'json'
    spacy = 'spacy'