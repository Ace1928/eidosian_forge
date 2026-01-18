from a parent namespace so that the forked namespace will contain everything
from kivy.event import EventDispatcher
from kivy.properties import StringProperty, ObjectProperty, AliasProperty
from kivy.context import register_context
def __knspace_clear_callbacks(self, *largs):
    for obj, name, uid in self.__callbacks:
        obj.unbind_uid(name, uid)
    last = self.__last_knspace
    self.__last_knspace = self.__callbacks = None
    assert self._knspace is None
    assert last
    new = self.__set_parent_knspace()
    if new is last:
        return
    self.property('_knspace').dispatch(self)
    name = self.knsname
    if not name:
        return
    if getattr(last, name) == self:
        setattr(last, name, None)
    if new:
        setattr(new, name, self)
    else:
        raise ValueError('Object has name "{}", but no namespace'.format(name))