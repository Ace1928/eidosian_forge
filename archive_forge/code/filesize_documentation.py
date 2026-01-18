from __future__ import absolute_import, division, print_function
import re
import os
import math
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
Compute dd options to grow or truncate a file.