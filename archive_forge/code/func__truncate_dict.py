import base64
import inspect
import logging
import socket
import subprocess
import uuid
from contextlib import closing
from itertools import islice
from sys import version_info
def _truncate_dict(d, max_key_length=None, max_value_length=None):
    """
    Truncates keys and/or values in a dictionary to the specified maximum length.
    Truncated items will be converted to strings and ellipsized.
    """
    key_is_none = max_key_length is None
    val_is_none = max_value_length is None
    if key_is_none and val_is_none:
        raise ValueError('Must specify at least either `max_key_length` or `max_value_length`')
    truncated = {}
    for k, v in d.items():
        should_truncate_key = not key_is_none and len(str(k)) > max_key_length
        should_truncate_val = not val_is_none and len(str(v)) > max_value_length
        new_k = _truncate_and_ellipsize(k, max_key_length) if should_truncate_key else k
        if should_truncate_key:
            msg = f'Truncated the key `{new_k}`'
            _logger.warning(msg)
        new_v = _truncate_and_ellipsize(v, max_value_length) if should_truncate_val else v
        if should_truncate_val:
            msg = f'Truncated the value of the key `{new_k}`. Truncated value: `{new_v}`'
            _logger.warning(msg)
        truncated[new_k] = new_v
    return truncated