import abc
import collections
import copy
import dataclasses
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from .arrow_reader import FileInstructions, make_file_instructions
from .naming import _split_re
from .utils.py_utils import NonMutableDict, asdict
class PercentSliceMeta(type):

    def __getitem__(cls, slice_value):
        if not isinstance(slice_value, slice):
            raise ValueError(f'datasets.percent should only be called with slice, not {slice_value}')
        return slice_value