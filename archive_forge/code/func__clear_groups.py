from kivy.properties import ObjectProperty, BooleanProperty
from kivy.uix.behaviors.button import ButtonBehavior
from weakref import ref
@staticmethod
def _clear_groups(wk):
    groups = ToggleButtonBehavior.__groups
    for group in list(groups.values()):
        if wk in group:
            group.remove(wk)
            break