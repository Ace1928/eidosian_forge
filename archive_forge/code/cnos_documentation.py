from __future__ import (absolute_import, division, print_function)
import re
import json
from ansible.errors import AnsibleConnectionFailure
from ansible.module_utils.common._collections_compat import Mapping
from ansible.module_utils._text import to_bytes, to_text
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list
from ansible.plugins.cliconf import CliconfBase, enable_mode

        Make sure we are in the operational cli mode
        :return: None
        