from __future__ import print_function, unicode_literals
import typing
import functools
import six
from six import text_type
class MissingInfoNamespace(AttributeError):
    """An expected namespace is missing."""

    def __init__(self, namespace):
        self.namespace = namespace
        msg = "namespace '{}' is required for this attribute"
        super(MissingInfoNamespace, self).__init__(msg.format(namespace))

    def __reduce__(self):
        return (type(self), (self.namespace,))