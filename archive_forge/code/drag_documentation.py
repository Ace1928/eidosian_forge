from kivy.clock import Clock
from kivy.properties import NumericProperty, ReferenceListProperty
from kivy.config import Config
from kivy.metrics import sp
from functools import partial
Position and size of the axis aligned bounding rectangle where dragging
    is allowed.

    :attr:`drag_rectangle` is a :class:`~kivy.properties.ReferenceListProperty`
    of (:attr:`drag_rect_x`, :attr:`drag_rect_y`, :attr:`drag_rect_width`,
    :attr:`drag_rect_height`) properties.
    