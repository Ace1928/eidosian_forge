from __future__ import (absolute_import, division, print_function)
import os
import re
from collections import namedtuple
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six.moves import shlex_quote
from ansible_collections.community.docker.plugins.module_utils.util import DockerBaseClass
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils._logfmt import (
def _extract_logfmt_event(line, warn_function=None):
    try:
        result = _parse_logfmt_line(line, logrus_mode=True)
    except _InvalidLogFmt:
        return (None, False)
    if 'time' not in result or 'level' not in result or 'msg' not in result:
        return (None, False)
    if result['level'] == 'warning':
        if warn_function:
            warn_function(result['msg'])
        return (None, True)
    return (None, False)