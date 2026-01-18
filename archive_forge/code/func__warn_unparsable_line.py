from __future__ import (absolute_import, division, print_function)
import os
import re
from collections import namedtuple
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six.moves import shlex_quote
from ansible_collections.community.docker.plugins.module_utils.util import DockerBaseClass
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils._logfmt import (
def _warn_unparsable_line(line, warn_function):
    if warn_function:
        warn_function('Cannot parse event from line: {0!r}. Please report this at https://github.com/ansible-collections/community.docker/issues/new?assignees=&labels=&projects=&template=bug_report.md'.format(line))