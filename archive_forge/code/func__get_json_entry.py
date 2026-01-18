import hashlib
import json
import logging
import os
from pathlib import Path
import pickle
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, Optional, Tuple, Union, cast
import pgzip
import torch
from torch import Tensor
from fairscale.internal.containers import from_np, to_np
from .utils import ExitCode
def _get_json_entry(d: Dict[str, Any]) -> Dict[str, Any]:
    """Get a dict from a json entry.

    This fills in any missing entries in case we load an older version
    json file from the disk.
    """
    for int_key_init_zero in [ENTRY_RF_KEY, ENTRY_OS_KEY, STORE_DS_KEY, ENTRY_CS_KEY]:
        if int_key_init_zero not in d.keys():
            d[int_key_init_zero] = 0
    for bool_key_init_false in [ENTRY_COMP_KEY]:
        if bool_key_init_false not in d.keys():
            d[bool_key_init_false] = False
    for dict_key_init_empty in [ENTRY_NAMES_KEY]:
        if dict_key_init_empty not in d.keys():
            d[dict_key_init_empty] = {}
    return d