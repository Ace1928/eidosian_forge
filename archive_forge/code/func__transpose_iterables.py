import os
import sys
import pickle
from collections import defaultdict
import re
from copy import deepcopy
from glob import glob
from pathlib import Path
from traceback import format_exception
from hashlib import sha1
from functools import reduce
import numpy as np
from ... import logging, config
from ...utils.filemanip import (
from ...utils.misc import str2bool
from ...utils.functions import create_function_from_source
from ...interfaces.base.traits_extension import (
from ...interfaces.base.support import Bunch, InterfaceResult
from ...interfaces.base import CommandLine
from ...interfaces.utility import IdentityInterface
from ...utils.provenance import ProvStore, pm, nipype_ns, get_id
from inspect import signature
def _transpose_iterables(fields, values):
    """
    Converts the given fields and tuple values into a standardized
    iterables value.

    If the input values is a synchronize iterables dictionary, then
    the result is a (field, {key: values}) list.

    Otherwise, the result is a list of (field: value list) pairs.
    """
    if isinstance(values, dict):
        transposed = dict([(field, defaultdict(list)) for field in fields])
        for key, tuples in list(values.items()):
            for kvals in tuples:
                for idx, val in enumerate(kvals):
                    if val is not None:
                        transposed[fields[idx]][key].append(val)
        return list(transposed.items())
    return list(zip(fields, [[v for v in list(transpose) if v is not None] for transpose in zip(*values)]))