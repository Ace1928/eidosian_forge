import numpy as np
from functools import wraps
from scipy._lib._docscrape import FunctionDoc, Parameter
from scipy._lib._util import _contains_nan, AxisError, _get_nan
import inspect
def _remove_sentinel(samples, paired, sentinel):
    """Remove sentinel values from paired or unpaired 1D samples"""
    if not paired:
        return [sample[sample != sentinel] for sample in samples]
    sentinels = samples[0] == sentinel
    for sample in samples[1:]:
        sentinels = sentinels | (sample == sentinel)
    not_sentinels = ~sentinels
    return [sample[not_sentinels] for sample in samples]