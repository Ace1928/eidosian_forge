from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six import string_types
from ansible_collections.community.dns.plugins.module_utils.http import (
def _split_text_namespace(node, text):
    i = text.find(':')
    if i < 0:
        return (text, None)
    ns = node.nsmap.get(text[:i])
    text = text[i + 1:]
    return (text, ns)