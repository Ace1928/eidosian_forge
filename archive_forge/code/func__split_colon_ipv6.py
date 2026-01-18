from __future__ import absolute_import, division, print_function
import abc
import os
import re
import shlex
from functools import partial
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.common.text.formatters import human_to_bytes
from ansible.module_utils.six import string_types
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils._platform import (
from ansible_collections.community.docker.plugins.module_utils._api.utils.utils import (
def _split_colon_ipv6(text, module):
    """
    Split string by ':', while keeping IPv6 addresses in square brackets in one component.
    """
    if '[' not in text:
        return text.split(':')
    start = 0
    result = []
    while start < len(text):
        i = text.find('[', start)
        if i < 0:
            result.extend(text[start:].split(':'))
            break
        j = text.find(']', i)
        if j < 0:
            module.fail_json(msg='Cannot find closing "]" in input "{0}" for opening "[" at index {1}!'.format(text, i + 1))
        result.extend(text[start:i].split(':'))
        k = text.find(':', j)
        if k < 0:
            result[-1] += text[i:]
            start = len(text)
        else:
            result[-1] += text[i:k]
            if k == len(text):
                result.append('')
                break
            start = k + 1
    return result