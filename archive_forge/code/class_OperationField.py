from __future__ import absolute_import, division, print_function
from ansible_collections.community.network.plugins.module_utils.network.ftd.common import HTTPMethod
from ansible.module_utils.six import integer_types, string_types, iteritems
class OperationField:
    URL = 'url'
    METHOD = 'method'
    PARAMETERS = 'parameters'
    MODEL_NAME = 'modelName'
    DESCRIPTION = 'description'
    RETURN_MULTIPLE_ITEMS = 'returnMultipleItems'
    TAGS = 'tags'