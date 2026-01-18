from __future__ import absolute_import, division, print_function
from ansible_collections.community.network.plugins.module_utils.network.ftd.common import HTTPMethod
from ansible.module_utils.six import integer_types, string_types, iteritems
class PropName:
    ENUM = 'enum'
    TYPE = 'type'
    REQUIRED = 'required'
    INVALID_TYPE = 'invalid_type'
    REF = '$ref'
    ALL_OF = 'allOf'
    BASE_PATH = 'basePath'
    PATHS = 'paths'
    OPERATION_ID = 'operationId'
    SCHEMA = 'schema'
    ITEMS = 'items'
    PROPERTIES = 'properties'
    RESPONSES = 'responses'
    NAME = 'name'
    DESCRIPTION = 'description'