from collections.abc import Iterable, Mapping
from ..pyutils.ordereddict import OrderedDict
from ..utils.assert_valid_name import assert_valid_name
from .definition import GraphQLArgument, GraphQLNonNull, is_input_type
from .scalars import GraphQLBoolean, GraphQLString
class DirectiveLocation(object):
    QUERY = 'QUERY'
    MUTATION = 'MUTATION'
    SUBSCRIPTION = 'SUBSCRIPTION'
    FIELD = 'FIELD'
    FRAGMENT_DEFINITION = 'FRAGMENT_DEFINITION'
    FRAGMENT_SPREAD = 'FRAGMENT_SPREAD'
    INLINE_FRAGMENT = 'INLINE_FRAGMENT'
    SCHEMA = 'SCHEMA'
    SCALAR = 'SCALAR'
    OBJECT = 'OBJECT'
    FIELD_DEFINITION = 'FIELD_DEFINITION'
    ARGUMENT_DEFINITION = 'ARGUMENT_DEFINITION'
    INTERFACE = 'INTERFACE'
    UNION = 'UNION'
    ENUM = 'ENUM'
    ENUM_VALUE = 'ENUM_VALUE'
    INPUT_OBJECT = 'INPUT_OBJECT'
    INPUT_FIELD_DEFINITION = 'INPUT_FIELD_DEFINITION'
    OPERATION_LOCATIONS = [QUERY, MUTATION, SUBSCRIPTION]
    FRAGMENT_LOCATIONS = [FRAGMENT_DEFINITION, FRAGMENT_SPREAD, INLINE_FRAGMENT]
    FIELD_LOCATIONS = [FIELD]