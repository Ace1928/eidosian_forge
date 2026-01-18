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
def _can_read_timestamps_from_packets(container: 'av.container.Container') -> bool:
    extradata = container.streams[0].codec_context.extradata
    if extradata is None:
        return False
    if b'Lavc' in extradata:
        return True
    return False