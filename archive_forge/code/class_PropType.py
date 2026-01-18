from __future__ import absolute_import, division, print_function
from ansible_collections.community.network.plugins.module_utils.network.ftd.common import HTTPMethod
from ansible.module_utils.six import integer_types, string_types, iteritems
class PropType:
    STRING = 'string'
    BOOLEAN = 'boolean'
    INTEGER = 'integer'
    NUMBER = 'number'
    OBJECT = 'object'
    ARRAY = 'array'
    FILE = 'file'