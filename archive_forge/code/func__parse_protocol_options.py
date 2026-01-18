from __future__ import absolute_import, division, print_function
import re
from collections import deque
from copy import deepcopy
from ansible.module_utils._text import to_text
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.argspec.acls.acls import (
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.utils.utils import (
def _parse_protocol_options(rendered_ace, ace_queue, protocol):
    """
            Parses the ACE queue and populates protocol specific options
            of the required dictionary and updates the ACE dictionary, i.e.,
            `rendered_ace`.

            :param rendered_ace: The dictionary containing the ACE in structured format
            :param ace_queue: The ACE queue
            :param protocol: Specifies the protocol that will be populated under
                             `protocol_options` dictionary
            """
    if len(ace_queue) > 0:
        protocol_options = {protocol: {}}
        for match_bit in PROTOCOL_OPTIONS.get(protocol, ()):
            if match_bit.replace('_', '-') in ace_queue:
                protocol_options[protocol][match_bit] = True
                ace_queue.remove(match_bit.replace('_', '-'))
        rendered_ace['protocol_options'] = protocol_options