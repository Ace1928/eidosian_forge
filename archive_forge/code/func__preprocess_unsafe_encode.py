from __future__ import (absolute_import, division, print_function)
import json
import datetime
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.six.moves.collections_abc import Mapping
from ansible.module_utils.common.collections import is_sequence
def _preprocess_unsafe_encode(value):
    """Recursively preprocess a data structure converting instances of ``AnsibleUnsafe``
    into their JSON dict representations

    Used in ``AnsibleJSONEncoder.iterencode``
    """
    if _is_unsafe(value):
        value = {'__ansible_unsafe': to_text(value._strip_unsafe(), errors='surrogate_or_strict', nonstring='strict')}
    elif is_sequence(value):
        value = [_preprocess_unsafe_encode(v) for v in value]
    elif isinstance(value, Mapping):
        value = dict(((k, _preprocess_unsafe_encode(v)) for k, v in value.items()))
    return value