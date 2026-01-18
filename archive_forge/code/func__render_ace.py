from __future__ import absolute_import, division, print_function
import re
from collections import deque
from copy import deepcopy
from ansible.module_utils._text import to_text
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.argspec.acls.acls import (
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.utils.utils import (
def _render_ace(self, ace):
    """
        Parses an Access Control Entry (ACE) and converts it
        into model spec

        :param ace: An ACE in device specific format
        :rtype: dictionary
        :returns: The ACE in structured format
        """

    def _parse_src_dest(rendered_ace, ace_queue, direction):
        """
            Parses the ACE queue and populates address, wildcard_bits,
            host or any keys in the source/destination dictionary of
            ace dictionary, i.e., `rendered_ace`.

            :param rendered_ace: The dictionary containing the ACE in structured format
            :param ace_queue: The ACE queue
            :param direction: Specifies whether to populate `source` or `destination`
                              dictionary
            """
        element = ace_queue.popleft()
        if element == 'host':
            rendered_ace[direction] = {'host': ace_queue.popleft()}
        elif element == 'net-group':
            rendered_ace[direction] = {'net_group': ace_queue.popleft()}
        elif element == 'port-group':
            rendered_ace[direction] = {'port_group': ace_queue.popleft()}
        elif element == 'any':
            rendered_ace[direction] = {'any': True}
        elif '/' in element:
            rendered_ace[direction] = {'prefix': element}
        elif isipaddress(to_text(element)):
            rendered_ace[direction] = {'address': element, 'wildcard_bits': ace_queue.popleft()}

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

    def _parse_match_options(rendered_ace, ace_queue):
        """
            Parses the ACE queue and populates remaining options in the ACE dictionary,
            i.e., `rendered_ace`

            :param rendered_ace: The dictionary containing the ACE in structured format
            :param ace_queue: The ACE queue
            """
        if len(ace_queue) > 0:
            copy_ace_queue = deepcopy(ace_queue)
            for element in copy_ace_queue:
                if element == 'precedence':
                    ace_queue.popleft()
                    rendered_ace['precedence'] = ace_queue.popleft()
                elif element == 'dscp':
                    ace_queue.popleft()
                    dscp = {}
                    operation = ace_queue.popleft()
                    if operation in ('eq', 'gt', 'neq', 'lt', 'range'):
                        if operation == 'range':
                            dscp['range'] = {'start': ace_queue.popleft(), 'end': ace_queue.popleft()}
                        else:
                            dscp[operation] = ace_queue.popleft()
                    else:
                        dscp['eq'] = operation
                    rendered_ace['dscp'] = dscp
                elif element in ('packet-length', 'ttl'):
                    ace_queue.popleft()
                    element_dict = {}
                    operation = ace_queue.popleft()
                    if operation == 'range':
                        element_dict['range'] = {'start': ace_queue.popleft(), 'end': ace_queue.popleft()}
                    else:
                        element_dict[operation] = ace_queue.popleft()
                    rendered_ace[element.replace('-', '_')] = element_dict
                elif element in ('log', 'log-input', 'fragments', 'icmp-off', 'capture', 'destopts', 'authen', 'routing', 'hop-by-hop'):
                    rendered_ace[element.replace('-', '_')] = True
                    ace_queue.remove(element)
                copy_ace_queue = deepcopy(ace_queue)
    rendered_ace = {}
    split_ace = ace.split()
    ace_queue = deque(split_ace)
    sequence = int(ace_queue.popleft())
    rendered_ace['sequence'] = sequence
    operation = ace_queue.popleft()
    if operation == 'remark':
        rendered_ace['remark'] = ' '.join(split_ace[2:])
    else:
        rendered_ace['grant'] = operation
        rendered_ace['protocol'] = ace_queue.popleft()
        _parse_src_dest(rendered_ace, ace_queue, direction='source')
        _parse_port_protocol(rendered_ace, ace_queue, direction='source')
        _parse_src_dest(rendered_ace, ace_queue, direction='destination')
        _parse_port_protocol(rendered_ace, ace_queue, direction='destination')
        _parse_protocol_options(rendered_ace, ace_queue, protocol=rendered_ace['protocol'])
        _parse_match_options(rendered_ace, ace_queue)
        if len(ace_queue) > 0:
            rendered_ace = {'sequence': sequence, 'line': ' '.join(split_ace[1:])}
    return utils.remove_empties(rendered_ace)