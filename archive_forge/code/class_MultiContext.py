import abc
from yaql.language import exceptions
from yaql.language import runner
from yaql.language import specs
from yaql.language import utils
class MultiContext(ContextBase):

    def __init__(self, context_list, convention=None):
        self._context_list = context_list
        if convention is None:
            convention = context_list[0].convention
        parents = tuple(filter(lambda t: t, map(lambda t: t.parent, context_list)))
        if not parents:
            super(MultiContext, self).__init__(None, convention)
        elif len(parents) == 1:
            super(MultiContext, self).__init__(parents[0], convention)
        else:
            super(MultiContext, self).__init__(MultiContext(parents), convention)

    def register_function(self, spec, *args, **kwargs):
        self._context_list[0].register_function(spec, *args, **kwargs)

    def get_data(self, name, default=None, ask_parent=True):
        for context in self._context_list:
            result = context.get_data(name, utils.NO_VALUE, False)
            if result is not utils.NO_VALUE:
                return result
        ctx = self.parent
        while ask_parent and ctx:
            result = ctx.get_data(name, utils.NO_VALUE, False)
            if result is utils.NO_VALUE:
                ctx = ctx.parent
            else:
                return result
        return default

    def __setitem__(self, name, value):
        self._context_list[0][name] = value

    def __delitem__(self, name):
        for context in self._context_list:
            del context[name]

    def create_child_context(self):
        return Context(self)

    def keys(self):
        prev_keys = set()
        for context in self._context_list:
            for key in context.keys():
                if key not in prev_keys:
                    prev_keys.add(key)
                    yield key

    def delete_function(self, spec):
        for context in self._context_list:
            context.delete_function(spec)

    def __contains__(self, item):
        for context in self._context_list:
            if item in context:
                return True
        return False

    def get_functions(self, name, predicate=None, use_convention=False):
        result = set()
        is_exclusive = False
        for context in self._context_list:
            funcs, exclusive = context.get_functions(name, predicate, use_convention)
            result.update(funcs)
            if exclusive:
                is_exclusive = True
        return (result, is_exclusive)