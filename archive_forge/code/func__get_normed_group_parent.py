from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def _get_normed_group_parent(self, parent):
    """ Converts parent dict information into a more easy to use form.

        :param parent: parent describing dict
        """
    if parent['id']:
        return (parent['id'], True)
    return (parent['name'], False)