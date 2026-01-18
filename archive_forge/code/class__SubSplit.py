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
class _SubSplit(SplitBase):
    """Represent a sub split of a split descriptor."""

    def __init__(self, split, slice_value):
        self._split = split
        self._slice_value = slice_value

    def get_read_instruction(self, split_dict):
        return self._split.get_read_instruction(split_dict)[self._slice_value]

    def __repr__(self):
        slice_str = '{start}:{stop}'
        if self._slice_value.step is not None:
            slice_str += ':{step}'
        slice_str = slice_str.format(start='' if self._slice_value.start is None else self._slice_value.start, stop='' if self._slice_value.stop is None else self._slice_value.stop, step=self._slice_value.step)
        return f'{repr(self._split)}(datasets.percent[{slice_str}])'