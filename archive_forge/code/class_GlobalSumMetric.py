import re
from abc import ABC, abstractmethod
from collections import Counter
import functools
import datetime
from typing import Union, List, Optional, Tuple, Set, Any, Dict
import torch
from parlai.core.message import Message
from parlai.utils.misc import warn_once
from parlai.utils.typing import TScalar, TVector
class GlobalSumMetric(GlobalMetric, SumMetric):
    """
    Global sum metric.

    Used for 'exs' and 'updates'.
    """
    pass