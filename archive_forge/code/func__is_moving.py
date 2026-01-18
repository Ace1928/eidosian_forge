from kivy.factory import Factory
from kivy.uix.button import Button
from kivy.properties import (OptionProperty, NumericProperty, ObjectProperty,
from kivy.uix.boxlayout import BoxLayout
@staticmethod
def _is_moving(sz_frm, diff, pos, minpos, maxpos):
    if sz_frm in ('l', 'b'):
        cmp = minpos
    else:
        cmp = maxpos
    if diff == 0:
        return False
    elif diff > 0 and pos <= cmp:
        return False
    elif diff < 0 and pos >= cmp:
        return False
    return True