from __future__ import (absolute_import, division, print_function)
import os
import re
from collections import namedtuple
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six.moves import shlex_quote
from ansible_collections.community.docker.plugins.module_utils.util import DockerBaseClass
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils._logfmt import (
def _extract_event(line, warn_function=None):
    match = _RE_RESOURCE_EVENT.match(line)
    if match is not None:
        status = match.group('status')
        msg = None
        if status not in DOCKER_STATUS:
            status, msg = (msg, status)
        return (Event(ResourceType.from_docker_compose_event(match.group('resource_type')), match.group('resource_id'), status, msg), True)
    match = _RE_PULL_EVENT.match(line)
    if match:
        return (Event(ResourceType.SERVICE, match.group('service'), match.group('status'), None), True)
    match = _RE_ERROR_EVENT.match(line)
    if match:
        return (Event(ResourceType.UNKNOWN, match.group('resource_id'), match.group('status'), match.group('msg') or None), True)
    match = _RE_WARNING_EVENT.match(line)
    if match:
        if warn_function:
            if match.group('msg'):
                msg = '{rid}: {msg}'
            else:
                msg = 'Unspecified warning for {rid}'
            warn_function(msg.format(rid=match.group('resource_id'), msg=match.group('msg')))
        return (None, True)
    match = _RE_PULL_PROGRESS.match(line)
    if match:
        return (Event(ResourceType.IMAGE_LAYER, match.group('layer'), match.group('status'), None), True)
    match = _RE_SKIPPED_EVENT.match(line)
    if match:
        return (Event(ResourceType.UNKNOWN, match.group('resource_id'), 'Skipped', match.group('msg')), True)
    match = _RE_BUILD_START_EVENT.match(line)
    if match:
        return (Event(ResourceType.SERVICE, match.group('resource_id'), 'Building', None), True)
    return (None, False)