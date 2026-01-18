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
class GlobalMetric:
    """
    A global metric is one that should not be aggregated across different tasks.

    Examples of global metric include things like learning rate and updates.
    These need to be accumulated or averaged over multiple parleys, but cannot
    be correlated with a single task.

    Key to it is the notion that any one worker or any one task already has a global
    view of the value, and so no combinations should be done. Note this is different
    then a FixedMetric, in that a GlobalMetric can be still averaged across multiple
    parleys(), but a FixedMetric is always fixed.
    """

    @property
    def is_global(self) -> bool:
        return True