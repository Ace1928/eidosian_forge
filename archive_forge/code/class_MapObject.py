from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
class MapObject(object):
    """Base class to wrap dict-like structures and support attributes for keys."""

    def __init__(self, props):
        self._props = props

    def __eq__(self, o):
        return self._props == o._props

    @property
    def props(self):
        return self._props

    def MakeSerializable(self):
        return self._props