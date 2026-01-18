from __future__ import absolute_import, division, print_function
import re
from collections import deque
from copy import deepcopy
from ansible.module_utils._text import to_text
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.argspec.acls.acls import (
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.utils.utils import (
def _parse_port_protocol(rendered_ace, ace_queue, direction):
    """
            Parses the ACE queue and populates `port_protocol` dictionary in the
            ACE dictionary, i.e., `rendered_ace`.

            :param rendered_ace: The dictionary containing the ACE in structured format
            :param ace_queue: The ACE queue
            :param direction: Specifies whether to populate `source` or `destination`
                              dictionary
            """
    if len(ace_queue) > 0 and ace_queue[0] in ('eq', 'gt', 'lt', 'neq', 'range'):
        element = ace_queue.popleft()
        port_protocol = {}
        if element == 'range':
            port_protocol['range'] = {'start': ace_queue.popleft(), 'end': ace_queue.popleft()}
        else:
            port_protocol[element] = ace_queue.popleft()
        if rendered_ace.get(direction):
            rendered_ace[direction]['port_protocol'] = port_protocol
        else:
            rendered_ace[direction] = {'port_protocol': port_protocol}