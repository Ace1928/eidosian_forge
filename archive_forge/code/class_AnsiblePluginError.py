from __future__ import (absolute_import, division, print_function)
import re
import traceback
from collections.abc import Sequence
from ansible.errors.yaml_strings import (
from ansible.module_utils.common.text.converters import to_native, to_text
class AnsiblePluginError(AnsibleError):
    """ base class for Ansible plugin-related errors that do not need AnsibleError contextual data """

    def __init__(self, message=None, plugin_load_context=None):
        super(AnsiblePluginError, self).__init__(message)
        self.plugin_load_context = plugin_load_context