from kivy.properties import OptionProperty, ObjectProperty, BooleanProperty, \
from kivy.config import Config
from kivy.base import EventLoop
def _get_focus_next(self, focus_dir):
    current = self
    walk_tree = 'walk' if focus_dir == 'focus_next' else 'walk_reverse'
    while 1:
        while getattr(current, focus_dir) is not None:
            current = getattr(current, focus_dir)
            if current is self or current is StopIteration:
                return None
            if current.is_focusable and (not current.disabled):
                return current
        itr = getattr(current, walk_tree)(loopback=True)
        if focus_dir == 'focus_next':
            next(itr)
        for current in itr:
            if isinstance(current, FocusBehavior):
                break
        if isinstance(current, FocusBehavior):
            if current is self:
                return None
            if current.is_focusable and (not current.disabled):
                return current
        else:
            return None