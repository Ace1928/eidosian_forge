from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
def SearchPatternErrorInLog(patterns, sc_log):
    for pattern in patterns:
        regex = re.compile(pattern, re.MULTILINE | re.IGNORECASE)
        if regex.search(sc_log):
            return True
    return False