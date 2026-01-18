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
class _SplitMerged(SplitBase):
    """Represent two split descriptors merged together."""

    def __init__(self, split1, split2):
        self._split1 = split1
        self._split2 = split2

    def get_read_instruction(self, split_dict):
        read_instruction1 = self._split1.get_read_instruction(split_dict)
        read_instruction2 = self._split2.get_read_instruction(split_dict)
        return read_instruction1 + read_instruction2

    def __repr__(self):
        return f'({repr(self._split1)} + {repr(self._split2)})'