import functools
import logging
import operator
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy
import wasabi.tables
from .. import util
from ..errors import Errors
from ..pipeline import MultiLabel_TextCategorizer, TextCategorizer
from ..training import Corpus
from ._util import Arg, Opt, app, import_code, setup_gpu
Filters provided config dictionary so that only the specified keys path remains.
        config (Dict[str, Any]): Configuration dictionary.
        keys (List[Any]): Path to value to set.
        full_key (str): Full user-specified key.
        RETURNS (Dict[str, Any]): Filtered dictionary.
        