from a parent namespace so that the forked namespace will contain everything
from kivy.event import EventDispatcher
from kivy.properties import StringProperty, ObjectProperty, AliasProperty
from kivy.context import register_context
def _set_knsname(self, value):
    old_name = self._knsname
    knspace = self.knspace
    if old_name and knspace and (getattr(knspace, old_name) == self):
        setattr(knspace, old_name, None)
    self._knsname = value
    if value:
        if knspace:
            setattr(knspace, value, self)
        else:
            raise ValueError('Object has name "{}", but no namespace'.format(value))