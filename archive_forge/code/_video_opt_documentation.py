import math
import warnings
from fractions import Fraction
from typing import Dict, List, Optional, Tuple, Union
import torch
from ..extension import _load_library

    Probe a video in memory and return VideoMetaData with info about the video
    This function is torchscriptable
    