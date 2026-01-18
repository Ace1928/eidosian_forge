from __future__ import (absolute_import, division, print_function)
from collections import defaultdict
import platform
import ansible.module_utils.compat.typing as t
from ansible.module_utils.facts import timeout
class CycleFoundInFactDeps(Exception):
    """Indicates there is a cycle in fact collector deps

    If collector-B requires collector-A, and collector-A requires
    collector-B, that is a cycle. In that case, there is no ordering
    that will satisfy B before A and A and before B. That will cause this
    error to be raised.
    """
    pass