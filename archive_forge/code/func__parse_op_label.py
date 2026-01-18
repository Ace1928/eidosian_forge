import collections
import copy
import json
import re
from tensorflow.python.platform import build_info
from tensorflow.python.platform import tf_logging as logging
def _parse_op_label(self, label):
    """Parses the fields in a node timeline label."""
    match = re.match('(.*) = (.*)\\((.*)\\)', label)
    if match is None:
        return ('unknown', 'unknown', [])
    nn, op, inputs = match.groups()
    if not inputs:
        inputs = []
    else:
        inputs = inputs.split(', ')
    return (nn, op, inputs)