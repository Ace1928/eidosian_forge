import os
import re
import sys
import uuid
import warnings
from collections import OrderedDict, defaultdict
from collections.abc import MutableMapping, Sequence
from copy import copy as ccopy
from copy import deepcopy
import datetime
from html import escape
from typing import (
import numpy as np
import xarray as xr
from packaging import version
from ..rcparams import rcParams
from ..utils import HtmlTemplate, _subset_list, _var_names, either_dict_or_kwargs
from .base import _extend_xr_method, _make_json_serializable, dict_to_dataset
@property
def _groups_all(self) -> List[str]:
    return self._groups + self._groups_warmup