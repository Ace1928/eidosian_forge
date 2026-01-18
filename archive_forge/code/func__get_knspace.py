from a parent namespace so that the forked namespace will contain everything
from kivy.event import EventDispatcher
from kivy.properties import StringProperty, ObjectProperty, AliasProperty
from kivy.context import register_context
def _get_knspace(self):
    _knspace = self._knspace
    if _knspace is not None:
        return _knspace
    if self.__callbacks is not None:
        return self.__last_knspace
    return self.__set_parent_knspace()