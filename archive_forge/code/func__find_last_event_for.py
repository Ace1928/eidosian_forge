from __future__ import (absolute_import, division, print_function)
import os
import re
from collections import namedtuple
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six.moves import shlex_quote
from ansible_collections.community.docker.plugins.module_utils.util import DockerBaseClass
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils._logfmt import (
def _find_last_event_for(events, resource_id):
    for index, event in enumerate(reversed(events)):
        if event.resource_id == resource_id:
            return (len(events) - 1 - index, event)
    return None