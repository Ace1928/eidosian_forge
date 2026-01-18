import os
import re
import tempfile
from oslo_concurrency import processutils
from oslo_log import log as logging
from oslo_utils.secretutils import md5
from os_brick import exception
from os_brick import executor
from os_brick.i18n import _
def _option_exists(self, options, opt_pattern):
    """Checks if the option exists in nfs options and returns position."""
    options = [x.strip() for x in options.split(',')] if options else []
    pos = 0
    for opt in options:
        pos = pos + 1
        if re.match(opt_pattern, opt, flags=0):
            return pos
    return 0