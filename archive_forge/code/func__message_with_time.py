import math
import numbers
import platform
import struct
import timeit
import warnings
from collections.abc import Sequence
from contextlib import contextmanager, suppress
from itertools import compress, islice
import numpy as np
from scipy.sparse import issparse
from .. import get_config
from ..exceptions import DataConversionWarning
from . import _joblib, metadata_routing
from ._bunch import Bunch
from ._estimator_html_repr import estimator_html_repr
from ._param_validation import Integral, Interval, validate_params
from .class_weight import compute_class_weight, compute_sample_weight
from .deprecation import deprecated
from .discovery import all_estimators
from .fixes import parse_version, threadpool_info
from .murmurhash import murmurhash3_32
from .validation import (
def _message_with_time(source, message, time):
    """Create one line message for logging purposes.

    Parameters
    ----------
    source : str
        String indicating the source or the reference of the message.

    message : str
        Short message.

    time : int
        Time in seconds.
    """
    start_message = '[%s] ' % source
    if time > 60:
        time_str = '%4.1fmin' % (time / 60)
    else:
        time_str = ' %5.1fs' % time
    end_message = ' %s, total=%s' % (message, time_str)
    dots_len = 70 - len(start_message) - len(end_message)
    return '%s%s%s' % (start_message, dots_len * '.', end_message)