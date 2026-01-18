import copy
from itertools import groupby
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Tuple, cast
import torch
from fairscale.nn.misc import FlattenParamsWrapper
Is x a dimensionless tensor?