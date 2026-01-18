from collections.abc import Iterable, Mapping
from ..pyutils.ordereddict import OrderedDict
from ..utils.assert_valid_name import assert_valid_name
from .definition import GraphQLArgument, GraphQLNonNull, is_input_type
from .scalars import GraphQLBoolean, GraphQLString
class GraphQLDirective(object):
    __slots__ = ('name', 'args', 'description', 'locations')

    def __init__(self, name, description=None, args=None, locations=None):
        assert name, 'Directive must be named.'
        assert_valid_name(name)
        assert isinstance(locations, Iterable), 'Must provide locations for directive.'
        self.name = name
        self.description = description
        self.locations = locations
        if args:
            assert isinstance(args, Mapping), '{} args must be a dict with argument names as keys.'.format(name)
            for arg_name, _arg in args.items():
                assert_valid_name(arg_name)
                assert is_input_type(_arg.type), '{}({}) argument type must be Input Type but got {}.'.format(name, arg_name, _arg.type)
        self.args = args or OrderedDict()