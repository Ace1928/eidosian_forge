from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
class Lag_interfacesTemplate(NetworkTemplate):

    def __init__(self, lines=None, module=None):
        super(Lag_interfacesTemplate, self).__init__(lines=lines, tmplt=self, module=module)
    PARSERS = [{'name': 'member', 'getval': re.compile('\n              ^interface\\s\n              (?P<member>\\S+)$', re.VERBOSE), 'setval': 'interface {{ member }}', 'result': {'{{ member }}': {'member': '{{ member }}'}}, 'shared': True}, {'name': 'channel', 'getval': re.compile('\n                \\s+channel-group\n                (\\s(?P<channel>\\d+))?\n                (\\smode\\s(?P<mode>active|passive|on|desirable|auto))?\n                (\\slink\\s(?P<link>\\d+))?\n                $', re.VERBOSE), 'setval': "channel-group{{ (' ' + channel|string) if channel is defined else '' }}{{ (' mode ' + mode) if mode is defined else '' }}{{ (' link ' + link|string) if link is defined else '' }}", 'result': {'{{ member }}': {'member': '{{ member }}', 'mode': '{{ mode }}', 'channel': 'Port-channel{{ channel }}', 'link': '{{ link }}'}}}]