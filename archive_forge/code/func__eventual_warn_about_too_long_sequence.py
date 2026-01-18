import copy
import json
import os
import re
import warnings
from collections import UserDict
from collections.abc import Mapping, Sized
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union
import numpy as np
from packaging import version
from . import __version__
from .dynamic_module_utils import custom_object_save
from .utils import (
def _eventual_warn_about_too_long_sequence(self, ids: List[int], max_length: Optional[int], verbose: bool):
    """
        Depending on the input and internal state we might trigger a warning about a sequence that is too long for its
        corresponding model

        Args:
            ids (`List[str]`): The ids produced by the tokenization
            max_length (`int`, *optional*): The max_length desired (does not trigger a warning if it is set)
            verbose (`bool`): Whether or not to print more information and warnings.

        """
    if max_length is None and len(ids) > self.model_max_length and verbose:
        if not self.deprecation_warnings.get('sequence-length-is-longer-than-the-specified-maximum', False):
            logger.warning(f'Token indices sequence length is longer than the specified maximum sequence length for this model ({len(ids)} > {self.model_max_length}). Running this sequence through the model will result in indexing errors')
        self.deprecation_warnings['sequence-length-is-longer-than-the-specified-maximum'] = True