from __future__ import (absolute_import, division, print_function)
import xml.etree.ElementTree as ET
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config
worker