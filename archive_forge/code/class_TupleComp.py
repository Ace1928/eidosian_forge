import sys
import os
import time
import marshal
import re
from enum import StrEnum, _simple_enum
from functools import cmp_to_key
from dataclasses import dataclass
from typing import Dict
class TupleComp:
    """This class provides a generic function for comparing any two tuples.
    Each instance records a list of tuple-indices (from most significant
    to least significant), and sort direction (ascending or descending) for
    each tuple-index.  The compare functions can then be used as the function
    argument to the system sort() function when a list of tuples need to be
    sorted in the instances order."""

    def __init__(self, comp_select_list):
        self.comp_select_list = comp_select_list

    def compare(self, left, right):
        for index, direction in self.comp_select_list:
            l = left[index]
            r = right[index]
            if l < r:
                return -direction
            if l > r:
                return direction
        return 0