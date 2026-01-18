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
def _validate_iterables(node, iterables, fields):
    """
    Raise TypeError if an iterables member is not iterable.

    Raise ValueError if an iterables member is not a (field, values) pair.

    Raise ValueError if an iterable field is not in the inputs.
    """
    if isinstance(iterables, dict):
        iterables = list(iterables.items())
    elif not isinstance(iterables, tuple) and (not isinstance(iterables, list)):
        raise ValueError('The %s iterables type is not a list or a dictionary: %s' % (node.name, iterables.__class__))
    for item in iterables:
        try:
            if len(item) != 2:
                raise ValueError('The %s iterables is not a [(field, values)] list' % node.name)
        except TypeError as e:
            raise TypeError('A %s iterables member is not iterable: %s' % (node.name, e))
        field, _ = item
        if field not in fields:
            raise ValueError('The %s iterables field is unrecognized: %s' % (node.name, field))