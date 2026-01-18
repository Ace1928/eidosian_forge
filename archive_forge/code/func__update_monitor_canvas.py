from kivy.uix.label import Label
from kivy.graphics import Rectangle, Color
from kivy.clock import Clock
from functools import partial
def _update_monitor_canvas(win, ctx, *largs):
    with win.canvas.after:
        ctx.overlay.pos = (0, win.height - 25)
        ctx.overlay.size = (win.width, 25)
        ctx.rectangle.pos = (5, win.height - 20)