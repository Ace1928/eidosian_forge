from a parent namespace so that the forked namespace will contain everything
from kivy.event import EventDispatcher
from kivy.properties import StringProperty, ObjectProperty, AliasProperty
from kivy.context import register_context
def __set_parent_knspace(self):
    callbacks = self.__callbacks = []
    fbind = self.fbind
    append = callbacks.append
    parent_key = self.knspace_key
    clear = self.__knspace_clear_callbacks
    append((self, 'knspace_key', fbind('knspace_key', clear)))
    if not parent_key:
        self.__last_knspace = knspace
        return knspace
    append((self, parent_key, fbind(parent_key, clear)))
    parent = getattr(self, parent_key, None)
    while parent is not None:
        fbind = parent.fbind
        parent_knspace = getattr(parent, 'knspace', 0)
        if parent_knspace != 0:
            append((parent, 'knspace', fbind('knspace', clear)))
            self.__last_knspace = parent_knspace
            return parent_knspace
        append((parent, parent_key, fbind(parent_key, clear)))
        new_parent = getattr(parent, parent_key, None)
        if new_parent is parent:
            break
        parent = new_parent
    self.__last_knspace = knspace
    return knspace