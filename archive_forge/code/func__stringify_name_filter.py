from __future__ import absolute_import, division, print_function
import copy
from functools import partial
from ansible_collections.community.network.plugins.module_utils.network.ftd.common import HTTPMethod, equal_objects, FtdConfigurationError, \
from ansible_collections.community.network.plugins.module_utils.network.ftd.fdm_swagger_client import OperationField, ValidationError
from ansible.module_utils.six import iteritems
def _stringify_name_filter(self, filters):
    build_version = self.get_build_version()
    if build_version >= '6.4.0':
        return 'fts~%s' % filters['name']
    return 'name:%s' % filters['name']