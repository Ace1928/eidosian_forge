import os
import hashlib
from typing import Any, Dict, List, Optional
from ansible.module_utils.six import iteritems, string_types
from ansible_collections.kubernetes.core.plugins.module_utils.args_common import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.core import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.exceptions import (
class unique_string(str):
    _low = None

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    def lower(self):
        if self._low is None:
            lower = str.lower(self)
            if str.__eq__(lower, self):
                self._low = self
            else:
                self._low = unique_string(lower)
        return self._low