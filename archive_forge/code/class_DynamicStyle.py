from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from six import with_metaclass
class DynamicStyle(Style):
    """
    Style class that can dynamically returns an other Style.

    :param get_style: Callable that returns a :class:`.Style` instance.
    """

    def __init__(self, get_style):
        self.get_style = get_style

    def get_attrs_for_token(self, token):
        style = self.get_style()
        assert isinstance(style, Style)
        return style.get_attrs_for_token(token)

    def invalidation_hash(self):
        return self.get_style().invalidation_hash()