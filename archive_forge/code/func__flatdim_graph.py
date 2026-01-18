import operator as op
from collections import OrderedDict
from functools import reduce, singledispatch
from typing import Dict as TypingDict
from typing import TypeVar, Union, cast
import numpy as np
from gym.spaces import (
@flatdim.register(Graph)
def _flatdim_graph(space: Graph):
    raise ValueError('Cannot get flattened size as the Graph Space in Gym has a dynamic size.')