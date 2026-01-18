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
class SplitReadInstruction:
    """Object containing the reading instruction for the dataset.

    Similarly to `SplitDescriptor` nodes, this object can be composed with itself,
    but the resolution happens instantaneously, instead of keeping track of the
    tree, such as all instructions are compiled and flattened in a single
    SplitReadInstruction object containing the list of files and slice to use.

    Once resolved, the instructions can be accessed with:

    ```
    read_instructions.get_list_sliced_split_info()  # List of splits to use
    ```

    """

    def __init__(self, split_info=None):
        self._splits = NonMutableDict(error_msg='Overlap between splits. Split {key} has been added with itself.')
        if split_info:
            self.add(SlicedSplitInfo(split_info=split_info, slice_value=None))

    def add(self, sliced_split):
        """Add a SlicedSplitInfo the read instructions."""
        self._splits[sliced_split.split_info.name] = sliced_split

    def __add__(self, other):
        """Merging split together."""
        split_instruction = SplitReadInstruction()
        split_instruction._splits.update(self._splits)
        split_instruction._splits.update(other._splits)
        return split_instruction

    def __getitem__(self, slice_value):
        """Sub-splits."""
        split_instruction = SplitReadInstruction()
        for v in self._splits.values():
            if v.slice_value is not None:
                raise ValueError(f'Trying to slice Split {v.split_info.name} which has already been sliced')
            v = v._asdict()
            v['slice_value'] = slice_value
            split_instruction.add(SlicedSplitInfo(**v))
        return split_instruction

    def get_list_sliced_split_info(self):
        return list(self._splits.values())