import functools
from wandb_promise import Promise, is_thenable, promise_for_dict
from ...pyutils.cached_property import cached_property
from ...pyutils.default_ordered_dict import DefaultOrderedDict
from ...type import (GraphQLInterfaceType, GraphQLList, GraphQLNonNull,
from ..base import ResolveInfo, Undefined, collect_fields, get_field_def
from ..values import get_argument_values
from ...error import GraphQLError
class AbstractFragment(object):

    def __init__(self, abstract_type, field_asts, context=None, info=None):
        self.abstract_type = abstract_type
        self.field_asts = field_asts
        self.context = context
        self.info = info
        self._fragments = {}

    @cached_property
    def possible_types(self):
        return self.context.schema.get_possible_types(self.abstract_type)

    @cached_property
    def possible_types_with_is_type_of(self):
        return [(type, type.is_type_of) for type in self.possible_types if callable(type.is_type_of)]

    def get_fragment(self, type):
        if isinstance(type, str):
            type = self.context.schema.get_type(type)
        if type not in self._fragments:
            assert type in self.possible_types, 'Runtime Object type "{}" is not a possible type for "{}".'.format(type, self.abstract_type)
            self._fragments[type] = Fragment(type, self.field_asts, self.context, self.info)
        return self._fragments[type]

    def resolve_type(self, result):
        return_type = self.abstract_type
        context = self.context.context_value
        if return_type.resolve_type:
            return return_type.resolve_type(result, context, self.info)
        for type, is_type_of in self.possible_types_with_is_type_of:
            if is_type_of(result, context, self.info):
                return type

    def resolve(self, root):
        _type = self.resolve_type(root)
        fragment = self.get_fragment(_type)
        return fragment.resolve(root)