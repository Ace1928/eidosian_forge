from collections import deque
import os
import six
from genshi.compat import numeric_types, StringIO, BytesIO
from genshi.core import Attrs, Stream, StreamEventKind, START, TEXT, _ensure
from genshi.input import ParseError
class Context(object):
    """Container for template input data.
    
    A context provides a stack of scopes (represented by dictionaries).
    
    Template directives such as loops can push a new scope on the stack with
    data that should only be available inside the loop. When the loop
    terminates, that scope can get popped off the stack again.
    
    >>> ctxt = Context(one='foo', other=1)
    >>> ctxt.get('one')
    'foo'
    >>> ctxt.get('other')
    1
    >>> ctxt.push(dict(one='frost'))
    >>> ctxt.get('one')
    'frost'
    >>> ctxt.get('other')
    1
    >>> ctxt.pop()
    {'one': 'frost'}
    >>> ctxt.get('one')
    'foo'
    """

    def __init__(self, **data):
        """Initialize the template context with the given keyword arguments as
        data.
        """
        self.frames = deque([data])
        self.pop = self.frames.popleft
        self.push = self.frames.appendleft
        self._match_templates = []
        self._choice_stack = []

        def defined(name):
            """Return whether a variable with the specified name exists in the
            expression scope."""
            return name in self

        def value_of(name, default=None):
            """If a variable of the specified name is defined, return its value.
            Otherwise, return the provided default value, or ``None``."""
            return self.get(name, default)
        data.setdefault('defined', defined)
        data.setdefault('value_of', value_of)

    def __repr__(self):
        return repr(list(self.frames))

    def __contains__(self, key):
        """Return whether a variable exists in any of the scopes.
        
        :param key: the name of the variable
        """
        return self._find(key)[1] is not None
    has_key = __contains__

    def __delitem__(self, key):
        """Remove a variable from all scopes.
        
        :param key: the name of the variable
        """
        for frame in self.frames:
            if key in frame:
                del frame[key]

    def __getitem__(self, key):
        """Get a variables's value, starting at the current scope and going
        upward.
        
        :param key: the name of the variable
        :return: the variable value
        :raises KeyError: if the requested variable wasn't found in any scope
        """
        value, frame = self._find(key)
        if frame is None:
            raise KeyError(key)
        return value

    def __len__(self):
        """Return the number of distinctly named variables in the context.
        
        :return: the number of variables in the context
        """
        return len(self.items())

    def __setitem__(self, key, value):
        """Set a variable in the current scope.
        
        :param key: the name of the variable
        :param value: the variable value
        """
        self.frames[0][key] = value

    def _find(self, key, default=None):
        """Retrieve a given variable's value and the frame it was found in.

        Intended primarily for internal use by directives.
        
        :param key: the name of the variable
        :param default: the default value to return when the variable is not
                        found
        """
        for frame in self.frames:
            if key in frame:
                return (frame[key], frame)
        return (default, None)

    def get(self, key, default=None):
        """Get a variable's value, starting at the current scope and going
        upward.
        
        :param key: the name of the variable
        :param default: the default value to return when the variable is not
                        found
        """
        for frame in self.frames:
            if key in frame:
                return frame[key]
        return default

    def keys(self):
        """Return the name of all variables in the context.
        
        :return: a list of variable names
        """
        keys = []
        for frame in self.frames:
            keys += [key for key in frame if key not in keys]
        return keys

    def items(self):
        """Return a list of ``(name, value)`` tuples for all variables in the
        context.
        
        :return: a list of variables
        """
        return [(key, self.get(key)) for key in self.keys()]

    def update(self, mapping):
        """Update the context from the mapping provided."""
        self.frames[0].update(mapping)

    def push(self, data):
        """Push a new scope on the stack.
        
        :param data: the data dictionary to push on the context stack.
        """

    def pop(self):
        """Pop the top-most scope from the stack."""

    def copy(self):
        """Create a copy of this Context object."""
        ctxt = Context()
        ctxt.frames.pop()
        ctxt.frames.extend(self.frames)
        ctxt._match_templates.extend(self._match_templates)
        ctxt._choice_stack.extend(self._choice_stack)
        return ctxt