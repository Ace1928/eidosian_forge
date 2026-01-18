from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.common.text.converters import to_bytes
import re
import os
def __filesystem_matches_criteria(self, filesystem, criteria):
    return (criteria['uuid'] is None or filesystem.uuid == criteria['uuid']) and (criteria['label'] is None or filesystem.label == criteria['label']) and (criteria['device'] is None or filesystem.contains_device(criteria['device']))