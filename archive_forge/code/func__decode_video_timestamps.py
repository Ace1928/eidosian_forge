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
def _decode_video_timestamps(container: 'av.container.Container') -> List[int]:
    if _can_read_timestamps_from_packets(container):
        return [x.pts for x in container.demux(video=0) if x.pts is not None]
    else:
        return [x.pts for x in container.decode(video=0) if x.pts is not None]