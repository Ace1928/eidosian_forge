import re
from . import errors
class LazyRegex:
    """A proxy around a real regex, which won't be compiled until accessed."""
    _regex_attributes_to_copy = ['__copy__', '__deepcopy__', 'findall', 'finditer', 'match', 'scanner', 'search', 'split', 'sub', 'subn']
    __slots__ = ['_real_regex', '_regex_args', '_regex_kwargs'] + _regex_attributes_to_copy

    def __init__(self, args, kwargs):
        """Create a new proxy object, passing in the args to pass to re.compile

        :param args: The `*args` to pass to re.compile
        :param kwargs: The `**kwargs` to pass to re.compile
        """
        self._real_regex = None
        self._regex_args = args
        self._regex_kwargs = kwargs

    def _compile_and_collapse(self):
        """Actually compile the requested regex"""
        self._real_regex = self._real_re_compile(*self._regex_args, **self._regex_kwargs)
        for attr in self._regex_attributes_to_copy:
            setattr(self, attr, getattr(self._real_regex, attr))

    def _real_re_compile(self, *args, **kwargs):
        """Thunk over to the original re.compile"""
        try:
            return re.compile(*args, **kwargs)
        except re.error as e:
            raise InvalidPattern('"' + args[0] + '" ' + str(e))

    def __getstate__(self):
        """Return the state to use when pickling."""
        return {'args': self._regex_args, 'kwargs': self._regex_kwargs}

    def __setstate__(self, dict):
        """Restore from a pickled state."""
        self._real_regex = None
        setattr(self, '_regex_args', dict['args'])
        setattr(self, '_regex_kwargs', dict['kwargs'])

    def __getattr__(self, attr):
        """Return a member from the proxied regex object.

        If the regex hasn't been compiled yet, compile it
        """
        if self._real_regex is None:
            self._compile_and_collapse()
        return getattr(self._real_regex, attr)