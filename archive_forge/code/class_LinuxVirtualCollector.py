from __future__ import (absolute_import, division, print_function)
import glob
import os
import re
from ansible.module_utils.facts.virtual.base import Virtual, VirtualCollector
from ansible.module_utils.facts.utils import get_file_content, get_file_lines
class LinuxVirtualCollector(VirtualCollector):
    _fact_class = LinuxVirtual
    _platform = 'Linux'