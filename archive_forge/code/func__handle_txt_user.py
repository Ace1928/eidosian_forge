from __future__ import (absolute_import, division, print_function)
import warnings
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.six import raise_from
from ansible_collections.community.dns.plugins.module_utils.record import (
from ansible_collections.community.dns.plugins.module_utils.conversion.base import (
from ansible_collections.community.dns.plugins.module_utils.conversion.txt import (
def _handle_txt_user(self, to_user, record):
    """
        Handle TXT records for sending to/from the user.
        """
    if self._txt_transformation == 'api':
        return
    if self._txt_transformation == 'quoted':
        if to_user:
            record.target = encode_txt_value(record.target, character_encoding=self._txt_character_encoding)
        else:
            record.target = decode_txt_value(record.target, character_encoding=self._txt_character_encoding)