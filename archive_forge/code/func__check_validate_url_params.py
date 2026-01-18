from __future__ import absolute_import, division, print_function
from ansible_collections.community.network.plugins.module_utils.network.ftd.common import HTTPMethod
from ansible.module_utils.six import integer_types, string_types, iteritems
def _check_validate_url_params(self, operation, params):
    if not operation or not isinstance(operation, string_types):
        raise IllegalArgumentException('The operation_name parameter must be a non-empty string')
    if not isinstance(params, dict):
        raise IllegalArgumentException('The params parameter must be a dict')
    if operation not in self._operations:
        raise IllegalArgumentException('{0} operation does not support'.format(operation))