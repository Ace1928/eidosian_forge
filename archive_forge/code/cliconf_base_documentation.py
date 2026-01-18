from __future__ import absolute_import, division, print_function
from abc import abstractmethod
from functools import wraps
from ansible.errors import AnsibleConnectionFailure, AnsibleError
from ansible.module_utils._text import to_bytes, to_text
from ansible.module_utils.common._collections_compat import Mapping
from ansible.plugins.cliconf import CliconfBase as CliconfBaseBase
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list

        Update the cli prompt context to ensure it is in operational mode
        :param config_context: It is string value to identify if the current cli prompt ends with config mode prompt
        :param exit_command: Command to execute to exit the config mode
        :return: None
        