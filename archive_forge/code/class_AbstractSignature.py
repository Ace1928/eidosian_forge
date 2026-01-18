from inspect import Parameter
from jedi.cache import memoize_method
from jedi import debug
from jedi import parser_utils
class AbstractSignature(_SignatureMixin):

    def __init__(self, value, is_bound=False):
        self.value = value
        self.is_bound = is_bound

    @property
    def name(self):
        return self.value.name

    @property
    def annotation_string(self):
        return ''

    def get_param_names(self, resolve_stars=False):
        param_names = self._function_value.get_param_names()
        if self.is_bound:
            return param_names[1:]
        return param_names

    def bind(self, value):
        raise NotImplementedError

    def matches_signature(self, arguments):
        return True

    def __repr__(self):
        if self.value is self._function_value:
            return '<%s: %s>' % (self.__class__.__name__, self.value)
        return '<%s: %s, %s>' % (self.__class__.__name__, self.value, self._function_value)