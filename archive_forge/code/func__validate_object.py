from __future__ import absolute_import, division, print_function
from ansible_collections.community.network.plugins.module_utils.network.ftd.common import HTTPMethod
from ansible.module_utils.six import integer_types, string_types, iteritems
def _validate_object(self, status, model, data, path):
    if self._is_enum(model):
        self._check_enum(status, model, data, path)
    elif self._is_object(model):
        self._check_object(status, model, data, path)