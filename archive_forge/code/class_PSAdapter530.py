from __future__ import (absolute_import, division, print_function)
import abc
import re
from os.path import basename
from ansible.module_utils import six
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils import deps
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
class PSAdapter530(PSAdapter):

    def __init__(self, psutil):
        super(PSAdapter530, self).__init__(psutil)

    def _process_iter(self, *attrs):
        return self._psutil.process_iter(attrs=attrs)

    @staticmethod
    def _get_attribute_from_proc(proc, attribute):
        return proc.info[attribute]