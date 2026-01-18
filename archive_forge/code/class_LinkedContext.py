import abc
from yaql.language import exceptions
from yaql.language import runner
from yaql.language import specs
from yaql.language import utils
class LinkedContext(ContextBase):
    """Context that is as a proxy to another context but has its own parent."""

    def __init__(self, parent_context, linked_context, convention=None):
        self.linked_context = linked_context
        if linked_context.parent:
            super(LinkedContext, self).__init__(LinkedContext(parent_context, linked_context.parent, convention), convention)
        else:
            super(LinkedContext, self).__init__(parent_context, convention)

    def register_function(self, spec, *args, **kwargs):
        return self.linked_context.register_function(spec, *args, **kwargs)

    def keys(self):
        return self.linked_context.keys()

    def get_data(self, name, default=None, ask_parent=True):
        result = self.linked_context.get_data(name, default=utils.NO_VALUE, ask_parent=False)
        if result is utils.NO_VALUE:
            if not ask_parent or not self.parent:
                return default
            return self.parent.get_data(name, default=default, ask_parent=True)
        return result

    def get_functions(self, name, predicate=None, use_convention=False):
        return self.linked_context.get_functions(name, predicate=predicate, use_convention=use_convention)

    def delete_function(self, spec):
        return self.linked_context.delete_function(spec)

    def __contains__(self, item):
        return item in self.linked_context

    def __delitem__(self, name):
        del self.linked_context[name]

    def __setitem__(self, name, value):
        self.linked_context[name] = value

    def create_child_context(self):
        return type(self.linked_context)(self)