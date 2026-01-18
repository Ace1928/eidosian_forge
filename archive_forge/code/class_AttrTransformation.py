import re
import sys
import six
from genshi.builder import Element
from genshi.core import Stream, Attrs, QName, TEXT, START, END, _ensure, Markup
from genshi.path import Path
class AttrTransformation(object):
    """Set an attribute on selected elements."""

    def __init__(self, name, value):
        """Construct transform.

        :param name: name of the attribute that should be set
        :param value: the value to set
        """
        self.name = name
        self.value = value

    def __call__(self, stream):
        """Apply the transform filter to the marked stream.

        :param stream: The marked event stream to filter
        """
        callable_value = hasattr(self.value, '__call__')
        for mark, (kind, data, pos) in stream:
            if mark is ENTER:
                if callable_value:
                    value = self.value(self.name, (kind, data, pos))
                else:
                    value = self.value
                if value is None:
                    attrs = data[1] - [QName(self.name)]
                else:
                    attrs = data[1] | [(QName(self.name), value)]
                data = (data[0], attrs)
            yield (mark, (kind, data, pos))