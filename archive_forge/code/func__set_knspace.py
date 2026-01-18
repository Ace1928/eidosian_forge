from a parent namespace so that the forked namespace will contain everything
from kivy.event import EventDispatcher
from kivy.properties import StringProperty, ObjectProperty, AliasProperty
from kivy.context import register_context
def _set_knspace(self, value):
    if value is self._knspace:
        return
    knspace = self._knspace or self.__last_knspace
    name = self.knsname
    if name and knspace and (getattr(knspace, name) == self):
        setattr(knspace, name, None)
    if value == 'fork':
        if not knspace:
            knspace = self.knspace
        if knspace:
            value = knspace.fork()
        else:
            raise ValueError('Cannot fork with no namespace')
    for obj, prop_name, uid in self.__callbacks or []:
        obj.unbind_uid(prop_name, uid)
    self.__last_knspace = self.__callbacks = None
    if name:
        if value is None:
            knspace = self.__set_parent_knspace()
            if knspace:
                setattr(knspace, name, self)
            self._knspace = None
        else:
            setattr(value, name, self)
            knspace = self._knspace = value
        if not knspace:
            raise ValueError('Object has name "{}", but no namespace'.format(name))
    else:
        if value is None:
            self.__set_parent_knspace()
        self._knspace = value