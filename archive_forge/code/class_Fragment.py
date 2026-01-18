import functools
from wandb_promise import Promise, is_thenable, promise_for_dict
from ...pyutils.cached_property import cached_property
from ...pyutils.default_ordered_dict import DefaultOrderedDict
from ...type import (GraphQLInterfaceType, GraphQLList, GraphQLNonNull,
from ..base import ResolveInfo, Undefined, collect_fields, get_field_def
from ..values import get_argument_values
from ...error import GraphQLError
class Fragment(object):

    def __init__(self, type, field_asts, context=None, info=None):
        self.type = type
        self.field_asts = field_asts
        self.context = context
        self.info = info

    @cached_property
    def partial_resolvers(self):
        return list(get_resolvers(self.context, self.type, self.field_asts))

    @cached_property
    def fragment_container(self):
        try:
            fields = next(zip(*self.partial_resolvers))
        except StopIteration:
            fields = tuple()

        class FragmentInstance(dict):
            set = dict.__setitem__

            def __iter__(self):
                return iter(fields)
        return FragmentInstance

    def have_type(self, root):
        return not self.type.is_type_of or self.type.is_type_of(root, self.context.context_value, self.info)

    def resolve(self, root):
        if root and (not self.have_type(root)):
            raise GraphQLError(u'Expected value of type "{}" but got: {}.'.format(self.type, type(root).__name__), self.info.field_asts)
        contains_promise = False
        final_results = self.fragment_container()
        for response_name, field_resolver in self.partial_resolvers:
            result = field_resolver.execute(root)
            if result is Undefined:
                continue
            if not contains_promise and is_thenable(result):
                contains_promise = True
            final_results[response_name] = result
        if not contains_promise:
            return final_results
        return promise_for_dict(final_results)

    def resolve_serially(self, root):

        def execute_field_callback(results, resolver):
            response_name, field_resolver = resolver
            result = field_resolver.execute(root)
            if result is Undefined:
                return results
            if is_thenable(result):

                def collect_result(resolved_result):
                    results[response_name] = resolved_result
                    return results
                return result.then(collect_result)
            results[response_name] = result
            return results

        def execute_field(prev_promise, resolver):
            return prev_promise.then(lambda results: execute_field_callback(results, resolver))
        return functools.reduce(execute_field, self.partial_resolvers, Promise.resolve(self.fragment_container()))

    def __eq__(self, other):
        return isinstance(other, Fragment) and (other.type == self.type and other.field_asts == self.field_asts and (other.context == self.context) and (other.info == self.info))