from __future__ import annotations
import itertools
import math
from collections.abc import Mapping, Iterable
from jinja2.filters import pass_environment
from ansible.errors import AnsibleFilterError, AnsibleFilterTypeError
from ansible.module_utils.common.text import formatters
from ansible.module_utils.six import binary_type, text_type
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.utils.display import Display
class FilterModule(object):
    """ Ansible math jinja2 filters """

    def filters(self):
        filters = {'log': logarithm, 'pow': power, 'root': inversepower, 'unique': unique, 'intersect': intersect, 'difference': difference, 'symmetric_difference': symmetric_difference, 'union': union, 'product': itertools.product, 'permutations': itertools.permutations, 'combinations': itertools.combinations, 'human_readable': human_readable, 'human_to_bytes': human_to_bytes, 'rekey_on_member': rekey_on_member, 'zip': zip, 'zip_longest': itertools.zip_longest}
        return filters