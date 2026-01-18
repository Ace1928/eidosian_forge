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
def _do_fail(e):
    if case_sensitive is False or attribute:
        raise AnsibleFilterError("Jinja2's unique filter failed and we cannot fall back to Ansible's version as it does not support the parameters supplied", orig_exc=e)