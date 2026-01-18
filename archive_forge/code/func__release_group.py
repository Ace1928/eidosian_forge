from kivy.properties import ObjectProperty, BooleanProperty
from kivy.uix.behaviors.button import ButtonBehavior
from weakref import ref
def _release_group(self, current):
    if self.group is None:
        return
    group = self.__groups[self.group]
    for item in group[:]:
        widget = item()
        if widget is None:
            group.remove(item)
        if widget is current:
            continue
        widget.state = 'normal'