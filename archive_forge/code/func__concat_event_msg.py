from __future__ import (absolute_import, division, print_function)
import os
import re
from collections import namedtuple
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six.moves import shlex_quote
from ansible_collections.community.docker.plugins.module_utils.util import DockerBaseClass
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils._logfmt import (
def _concat_event_msg(event, append_msg):
    return Event(event.resource_type, event.resource_id, event.status, '\n'.join((msg for msg in [event.msg, append_msg] if msg is not None)))