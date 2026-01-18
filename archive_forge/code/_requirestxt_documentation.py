import io
from collections import defaultdict
from itertools import filterfalse
from typing import Dict, List, Tuple, Mapping, TypeVar
from .. import _reqs
from ..extern.jaraco.text import yield_lines
from ..extern.packaging.requirements import Requirement
Given a Requirement, remove environment markers and return it