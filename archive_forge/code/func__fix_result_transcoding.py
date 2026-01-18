from collections import namedtuple
import functools
import re
import sys
import types
import warnings
import ipaddress
def _fix_result_transcoding():
    _result_pairs = ((DefragResult, DefragResultBytes), (SplitResult, SplitResultBytes), (ParseResult, ParseResultBytes))
    for _decoded, _encoded in _result_pairs:
        _decoded._encoded_counterpart = _encoded
        _encoded._decoded_counterpart = _decoded