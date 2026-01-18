from pprint import pformat
from .py3compat import MutableMapping
def __pretty_str__(self, nesting=1, indentation='    '):
    if self._value is NotImplemented:
        text = '<unread>'
    elif hasattr(self._value, '__pretty_str__'):
        text = self._value.__pretty_str__(nesting, indentation)
    else:
        text = str(self._value)
    return '%s: %s' % (self.__class__.__name__, text)