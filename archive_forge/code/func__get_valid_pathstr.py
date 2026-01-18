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
def _get_valid_pathstr(pathstr):
    """Remove disallowed characters from path

    Removes:  [][ (){}?:<>#!|"';]
    Replaces: ',' -> '.'
    """
    if not isinstance(pathstr, (str, bytes)):
        pathstr = str(pathstr)
    pathstr = pathstr.replace(os.sep, '..')
    pathstr = re.sub('[][ (){}?:<>#!|"\';]', '', pathstr)
    pathstr = pathstr.replace(',', '.')
    return pathstr