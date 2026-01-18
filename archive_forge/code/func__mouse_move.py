from kivy.core.image import Image
from kivy.graphics import Color, Rectangle
from kivy import kivy_data_dir
from kivy.compat import string_types
from os.path import join
from functools import partial
def _mouse_move(texture, size, offset, win, pos, *args):
    if hasattr(win, '_cursor'):
        c = win._cursor
    else:
        with win.canvas.after:
            Color(1, 1, 1, 1, mode='rgba')
            win._cursor = c = Rectangle(texture=texture, size=size)
    c.pos = (pos[0] + offset[0], pos[1] - size[1] + offset[1])