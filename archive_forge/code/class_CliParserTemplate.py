from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
class CliParserTemplate(NetworkTemplate):
    """The parser template base class"""

    def __init__(self, lines=None):
        super(CliParserTemplate, self).__init__(lines=lines, tmplt=self)