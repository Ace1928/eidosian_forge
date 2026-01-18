import abc
from yaql.language import exceptions
from yaql.language import runner
from yaql.language import specs
from yaql.language import utils
class ContextBase(metaclass=abc.ABCMeta):

    def __init__(self, parent_context=None, convention=None):
        self._parent_context = parent_context
        self._convention = convention
        if convention is None and parent_context:
            self._convention = parent_context.convention

    @property
    def parent(self):
        return self._parent_context

    @abc.abstractmethod
    def register_function(self, spec, *args, **kwargs):
        pass

    @abc.abstractmethod
    def get_data(self, name, default=None, ask_parent=True):
        return default

    def __getitem__(self, name):
        return self.get_data(name)

    @abc.abstractmethod
    def __setitem__(self, name, value):
        pass

    @abc.abstractmethod
    def __delitem__(self, name):
        pass

    @abc.abstractmethod
    def __contains__(self, item):
        return False

    def __call__(self, name, engine, receiver=utils.NO_VALUE, data_context=None, use_convention=False, function_filter=None):
        return lambda *args, **kwargs: runner.call(name, self, args, kwargs, engine, receiver, data_context, use_convention, function_filter)

    @abc.abstractmethod
    def get_functions(self, name, predicate=None, use_convention=False):
        return ([], False)

    @abc.abstractmethod
    def delete_function(self, spec):
        pass

    def collect_functions(self, name, predicate=None, use_convention=False):
        overloads = []
        p = self
        while p is not None:
            context_predicate = None
            if predicate:
                context_predicate = lambda fd: predicate(fd, p)
            layer_overloads, is_exclusive = p.get_functions(name, context_predicate, use_convention)
            p = None if is_exclusive else p.parent
            if layer_overloads:
                overloads.append(layer_overloads)
        return overloads

    def create_child_context(self):
        return type(self)(self)

    @property
    def convention(self):
        return self._convention

    @abc.abstractmethod
    def keys(self):
        return {}.keys()