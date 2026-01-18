from __future__ import absolute_import, division, print_function
import traceback
import re
import json
from itertools import chain
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils._text import to_native
from ansible.module_utils.common.collections import is_iterable
from ansible.module_utils.basic import AnsibleModule, missing_required_lib, _load_params
from ansible.module_utils.urls import open_url
def _change_choices_id(self, endpoint, data):
    """Used to change data that is static and under _choices for the application.
        ex. DEVICE_STATUS
        :returns data (dict): Returns the user defined data back with updated fields for _choices
        :params endpoint (str): The endpoint that will be used for mapping to required _choices
        :params data (dict): User defined data passed into the module
        """
    if REQUIRED_ID_FIND.get(endpoint):
        required_choices = REQUIRED_ID_FIND[endpoint]
        for choice in required_choices:
            if data.get(choice):
                if isinstance(data[choice], int):
                    continue
                choice_value = self._fetch_choice_value(data[choice], endpoint)
                data[choice] = choice_value
    return data