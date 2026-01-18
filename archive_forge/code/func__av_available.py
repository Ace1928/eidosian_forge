import gc
import math
import os
import re
import warnings
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from ..utils import _log_api_usage_once
from . import _video_opt
def _av_available() -> bool:
    return not isinstance(av, Exception)