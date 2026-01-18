from random import random
from kivy.event import EventDispatcher
from kivy.clock import Clock
from kivy.vector import Vector
from kivy.uix.floatlayout import FloatLayout
from kivy.graphics import Color, Line, Rectangle
from kivy.properties import (NumericProperty, BooleanProperty,
from colorsys import hsv_to_rgb
class GestureContainer(EventDispatcher):
    """Container object that stores information about a gesture. It has
    various properties that are updated by `GestureSurface` as drawing
    progresses.

    :Arguments:
        `touch`
            Touch object (as received by on_touch_down) used to initialize
            the gesture container. Required.

    :Properties:
        `active`
            Set to False once the gesture is complete (meets
            `max_stroke` setting or `GestureSurface.temporal_window`)

            :attr:`active` is a
            :class:`~kivy.properties.BooleanProperty`

        `active_strokes`
            Number of strokes currently active in the gesture, ie
            concurrent touches associated with this gesture.

            :attr:`active_strokes` is a
            :class:`~kivy.properties.NumericProperty`

        `max_strokes`
            Max number of strokes allowed in the gesture. This
            is set by `GestureSurface.max_strokes` but can
            be overridden for example from `on_gesture_start`.

            :attr:`max_strokes` is a
            :class:`~kivy.properties.NumericProperty`

        `was_merged`
            Indicates that this gesture has been merged with another
            gesture and should be considered discarded.

            :attr:`was_merged` is a
            :class:`~kivy.properties.BooleanProperty`

        `bbox`
            Dictionary with keys minx, miny, maxx, maxy. Represents the size
            of the gesture bounding box.

            :attr:`bbox` is a
            :class:`~kivy.properties.DictProperty`

        `width`
            Represents the width of the gesture.

            :attr:`width` is a
            :class:`~kivy.properties.NumericProperty`

        `height`
            Represents the height of the gesture.

            :attr:`height` is a
            :class:`~kivy.properties.NumericProperty`
    """
    active = BooleanProperty(True)
    active_strokes = NumericProperty(0)
    max_strokes = NumericProperty(0)
    was_merged = BooleanProperty(False)
    bbox = DictProperty({'minx': float('inf'), 'miny': float('inf'), 'maxx': float('-inf'), 'maxy': float('-inf')})
    width = NumericProperty(0)
    height = NumericProperty(0)

    def __init__(self, touch, **kwargs):
        self.color = kwargs.pop('color', [1.0, 1.0, 1.0])
        super(GestureContainer, self).__init__(**kwargs)
        self.id = str(touch.uid)
        self._create_time = Clock.get_time()
        self._update_time = None
        self._cleanup_time = None
        self._cache_time = 0
        self._vectors = None
        self._strokes = {}
        self.update_bbox(touch)

    def get_vectors(self, **kwargs):
        """Return strokes in a format that is acceptable for
        `kivy.multistroke.Recognizer` as a gesture candidate or template. The
        result is cached automatically; the cache is invalidated at the start
        and end of a stroke and if `update_bbox` is called. If you are going
        to analyze a gesture mid-stroke, you may need to set the `no_cache`
        argument to True."""
        if self._cache_time == self._update_time and (not kwargs.get('no_cache')):
            return self._vectors
        vecs = []
        append = vecs.append
        for tuid, l in self._strokes.items():
            lpts = l.points
            append([Vector(*pts) for pts in zip(lpts[::2], lpts[1::2])])
        self._vectors = vecs
        self._cache_time = self._update_time
        return vecs

    def handles(self, touch):
        """Returns True if this container handles the given touch"""
        if not self.active:
            return False
        return str(touch.uid) in self._strokes

    def accept_stroke(self, count=1):
        """Returns True if this container can accept `count` new strokes"""
        if not self.max_strokes:
            return True
        return len(self._strokes) + count <= self.max_strokes

    def update_bbox(self, touch):
        """Update gesture bbox from a touch coordinate"""
        x, y = (touch.x, touch.y)
        bb = self.bbox
        if x < bb['minx']:
            bb['minx'] = x
        if y < bb['miny']:
            bb['miny'] = y
        if x > bb['maxx']:
            bb['maxx'] = x
        if y > bb['maxy']:
            bb['maxy'] = y
        self.width = bb['maxx'] - bb['minx']
        self.height = bb['maxy'] - bb['miny']
        self._update_time = Clock.get_time()

    def add_stroke(self, touch, line):
        """Associate a list of points with a touch.uid; the line itself is
        created by the caller, but subsequent move/up events look it
        up via us. This is done to avoid problems during merge."""
        self._update_time = Clock.get_time()
        self._strokes[str(touch.uid)] = line
        self.active_strokes += 1

    def complete_stroke(self):
        """Called on touch up events to keep track of how many strokes
        are active in the gesture (we only want to dispatch event when
        the *last* stroke in the gesture is released)"""
        self._update_time = Clock.get_time()
        self.active_strokes -= 1

    def single_points_test(self):
        """Returns True if the gesture consists only of single-point strokes,
        we must discard it in this case, or an exception will be raised"""
        for tuid, l in self._strokes.items():
            if len(l.points) > 2:
                return False
        return True