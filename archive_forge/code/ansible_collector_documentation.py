from __future__ import (absolute_import, division, print_function)
import fnmatch
import sys
import ansible.module_utils.compat.typing as t
from ansible.module_utils.facts import timeout
from ansible.module_utils.facts import collector
from ansible.module_utils.common.collections import is_string
Collector that provides a facts with the gather_subset metadata.