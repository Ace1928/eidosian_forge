from random import random
from kivy.event import EventDispatcher
from kivy.clock import Clock
from kivy.vector import Vector
from kivy.uix.floatlayout import FloatLayout
from kivy.graphics import Color, Line, Rectangle
from kivy.properties import (NumericProperty, BooleanProperty,
from colorsys import hsv_to_rgb
class GestureSurface(FloatLayout):
    """Simple gesture surface to track/draw touch movements. Typically used
    to gather user input suitable for :class:`kivy.multistroke.Recognizer`.

    :Properties:
        `temporal_window`
            Time to wait from the last touch_up event before attempting
            to recognize the gesture. If you set this to 0, the
            `on_gesture_complete` event is not fired unless the
            :attr:`max_strokes` condition is met.

            :attr:`temporal_window` is a
            :class:`~kivy.properties.NumericProperty` and defaults to 2.0

        `max_strokes`
            Max number of strokes in a single gesture; if this is reached,
            recognition will start immediately on the final touch_up event.
            If this is set to 0, the `on_gesture_complete` event is not
            fired unless the :attr:`temporal_window` expires.

            :attr:`max_strokes` is a
            :class:`~kivy.properties.NumericProperty` and defaults to 2.0

        `bbox_margin`
            Bounding box margin for detecting gesture collisions, in
            pixels.

            :attr:`bbox_margin` is a
            :class:`~kivy.properties.NumericProperty` and defaults to 30

        `draw_timeout`
            Number of seconds to keep lines/bbox on canvas after the
            `on_gesture_complete` event is fired. If this is set to 0,
            gestures are immediately removed from the surface when
            complete.

            :attr:`draw_timeout` is a
            :class:`~kivy.properties.NumericProperty` and defaults to 3.0

        `color`
            Color used to draw the gesture, in RGB. This option does not
            have an effect if :attr:`use_random_color` is True.

            :attr:`color` is a
            :class:`~kivy.properties.ColorProperty` and defaults to
            [1, 1, 1, 1] (white)

            .. versionchanged:: 2.0.0
                Changed from :class:`~kivy.properties.ListProperty` to
                :class:`~kivy.properties.ColorProperty`.

        `use_random_color`
            Set to True to pick a random color for each gesture, if you do
            this then `color` is ignored. Defaults to False.

            :attr:`use_random_color` is a
            :class:`~kivy.properties.BooleanProperty` and defaults to False

        `line_width`
            Line width used for tracing touches on the surface. Set to 0
            if you only want to detect gestures without drawing anything.
            If you use 1.0, OpenGL GL_LINE is used for drawing; values > 1
            will use an internal drawing method based on triangles (less
            efficient), see :mod:`kivy.graphics`.

            :attr:`line_width` is a
            :class:`~kivy.properties.NumericProperty` and defaults to 2

        `draw_bbox`
            Set to True if you want to draw bounding box behind gestures.
            This only works if `line_width` >= 1. Default is False.

            :attr:`draw_bbox` is a
            :class:`~kivy.properties.BooleanProperty` and defaults to True

        `bbox_alpha`
            Opacity for bounding box if `draw_bbox` is True. Default 0.1

            :attr:`bbox_alpha` is a
            :class:`~kivy.properties.NumericProperty` and defaults to 0.1

    :Events:
        `on_gesture_start` :class:`GestureContainer`
            Fired when a new gesture is initiated on the surface, i.e. the
            first on_touch_down that does not collide with an existing
            gesture on the surface.

        `on_gesture_extend` :class:`GestureContainer`
            Fired when a touch_down event occurs within an existing gesture.

        `on_gesture_merge` :class:`GestureContainer`, :class:`GestureContainer`
            Fired when two gestures collide and get merged to one gesture.
            The first argument is the gesture that has been merged (no longer
            valid); the second is the combined (resulting) gesture.

        `on_gesture_complete` :class:`GestureContainer`
            Fired when a set of strokes is considered a complete gesture,
            this happens when `temporal_window` expires or `max_strokes`
            is reached. Typically you will bind to this event and use
            the provided `GestureContainer` get_vectors() method to
            match against your gesture database.

        `on_gesture_cleanup` :class:`GestureContainer`
            Fired `draw_timeout` seconds after `on_gesture_complete`,
            The gesture will be removed from the canvas (if line_width > 0 or
            draw_bbox is True) and the internal gesture list before this.

        `on_gesture_discard` :class:`GestureContainer`
            Fired when a gesture does not meet the minimum size requirements
            for recognition (width/height < 5, or consists only of single-
            point strokes).
    """
    temporal_window = NumericProperty(2.0)
    draw_timeout = NumericProperty(3.0)
    max_strokes = NumericProperty(4)
    bbox_margin = NumericProperty(30)
    line_width = NumericProperty(2)
    color = ColorProperty([1.0, 1.0, 1.0, 1.0])
    use_random_color = BooleanProperty(False)
    draw_bbox = BooleanProperty(False)
    bbox_alpha = NumericProperty(0.1)

    def __init__(self, **kwargs):
        super(GestureSurface, self).__init__(**kwargs)
        self._gestures = []
        self.register_event_type('on_gesture_start')
        self.register_event_type('on_gesture_extend')
        self.register_event_type('on_gesture_merge')
        self.register_event_type('on_gesture_complete')
        self.register_event_type('on_gesture_cleanup')
        self.register_event_type('on_gesture_discard')

    def on_touch_down(self, touch):
        """When a new touch is registered, the first thing we do is to test if
        it collides with the bounding box of another known gesture. If so, it
        is assumed to be part of that gesture.
        """
        if not self.collide_point(touch.x, touch.y):
            return
        touch.grab(self)
        g = self.find_colliding_gesture(touch)
        new = False
        if g is None:
            g = self.init_gesture(touch)
            new = True
        self.init_stroke(g, touch)
        if new:
            self.dispatch('on_gesture_start', g, touch)
        else:
            self.dispatch('on_gesture_extend', g, touch)
        return True

    def on_touch_move(self, touch):
        """When a touch moves, we add a point to the line on the canvas so the
        path is updated. We must also check if the new point collides with the
        bounding box of another gesture - if so, they should be merged."""
        if touch.grab_current is not self:
            return
        if not self.collide_point(touch.x, touch.y):
            return
        g = self.get_gesture(touch)
        collision = self.find_colliding_gesture(touch)
        if collision is not None and g.accept_stroke(len(collision._strokes)):
            merge = self.merge_gestures(g, collision)
            if g.was_merged:
                self.dispatch('on_gesture_merge', g, collision)
            else:
                self.dispatch('on_gesture_merge', collision, g)
            g = merge
        else:
            g.update_bbox(touch)
        g._strokes[str(touch.uid)].points += (touch.x, touch.y)
        if self.draw_bbox:
            self._update_canvas_bbox(g)
        return True

    def on_touch_up(self, touch):
        if touch.grab_current is not self:
            return
        touch.ungrab(self)
        g = self.get_gesture(touch)
        g.complete_stroke()
        if not g.accept_stroke():
            self._complete_dispatcher(0)
        elif self.temporal_window > 0:
            Clock.schedule_once(self._complete_dispatcher, self.temporal_window)

    def init_gesture(self, touch):
        """Create a new gesture from touch, i.e. it's the first on
        surface, or was not close enough to any existing gesture (yet)"""
        col = self.color
        if self.use_random_color:
            col = hsv_to_rgb(random(), 1.0, 1.0)
        g = GestureContainer(touch, max_strokes=self.max_strokes, color=col)
        if self.draw_bbox:
            bb = g.bbox
            with self.canvas:
                Color(col[0], col[1], col[2], self.bbox_alpha, mode='rgba', group=g.id)
                g._bbrect = Rectangle(group=g.id, pos=(bb['minx'], bb['miny']), size=(bb['maxx'] - bb['minx'], bb['maxy'] - bb['miny']))
        self._gestures.append(g)
        return g

    def init_stroke(self, g, touch):
        points = [touch.x, touch.y]
        col = g.color
        new_line = Line(points=points, width=self.line_width, group=g.id)
        g._strokes[str(touch.uid)] = new_line
        if self.line_width:
            canvas_add = self.canvas.add
            canvas_add(Color(col[0], col[1], col[2], mode='rgb', group=g.id))
            canvas_add(new_line)
        g.update_bbox(touch)
        if self.draw_bbox:
            self._update_canvas_bbox(g)
        g.add_stroke(touch, new_line)

    def get_gesture(self, touch):
        """Returns GestureContainer associated with given touch"""
        for g in self._gestures:
            if g.active and g.handles(touch):
                return g
        raise Exception('get_gesture() failed to identify ' + str(touch.uid))

    def find_colliding_gesture(self, touch):
        """Checks if a touch x/y collides with the bounding box of an existing
        gesture. If so, return it (otherwise returns None)
        """
        touch_x, touch_y = touch.pos
        for g in self._gestures:
            if g.active and (not g.handles(touch)) and g.accept_stroke():
                bb = g.bbox
                margin = self.bbox_margin
                minx = bb['minx'] - margin
                miny = bb['miny'] - margin
                maxx = bb['maxx'] + margin
                maxy = bb['maxy'] + margin
                if minx <= touch_x <= maxx and miny <= touch_y <= maxy:
                    return g
        return None

    def merge_gestures(self, g, other):
        """Merges two gestures together, the oldest one is retained and the
        newer one gets the `GestureContainer.was_merged` flag raised."""
        swap = other._create_time < g._create_time
        a = swap and other or g
        b = swap and g or other
        abbox = a.bbox
        bbbox = b.bbox
        if bbbox['minx'] < abbox['minx']:
            abbox['minx'] = bbbox['minx']
        if bbbox['miny'] < abbox['miny']:
            abbox['miny'] = bbbox['miny']
        if bbbox['maxx'] > abbox['maxx']:
            abbox['maxx'] = bbbox['maxx']
        if bbbox['maxy'] > abbox['maxy']:
            abbox['maxy'] = bbbox['maxy']
        astrokes = a._strokes
        lw = self.line_width
        a_id = a.id
        col = a.color
        self.canvas.remove_group(b.id)
        canv_add = self.canvas.add
        for uid, old in b._strokes.items():
            new_line = Line(points=old.points, width=old.width, group=a_id)
            astrokes[uid] = new_line
            if lw:
                canv_add(Color(col[0], col[1], col[2], mode='rgb', group=a_id))
                canv_add(new_line)
        b.active = False
        b.was_merged = True
        a.active_strokes += b.active_strokes
        a._update_time = Clock.get_time()
        return a

    def _update_canvas_bbox(self, g):
        if not hasattr(g, '_bbrect'):
            return
        bb = g.bbox
        g._bbrect.pos = (bb['minx'], bb['miny'])
        g._bbrect.size = (bb['maxx'] - bb['minx'], bb['maxy'] - bb['miny'])

    def _complete_dispatcher(self, dt):
        """This method is scheduled on all touch up events. It will dispatch
        the `on_gesture_complete` event for all completed gestures, and remove
        merged gestures from the internal gesture list."""
        need_cleanup = False
        gest = self._gestures
        timeout = self.draw_timeout
        twin = self.temporal_window
        get_time = Clock.get_time
        for idx, g in enumerate(gest):
            if g.was_merged:
                del gest[idx]
                continue
            if not g.active or g.active_strokes != 0:
                continue
            t1 = g._update_time + twin
            t2 = get_time() + UNDERSHOOT_MARGIN
            if not g.accept_stroke() or t1 <= t2:
                discard = False
                if g.width < 5 and g.height < 5:
                    discard = True
                elif g.single_points_test():
                    discard = True
                need_cleanup = True
                g.active = False
                g._cleanup_time = get_time() + timeout
                if discard:
                    self.dispatch('on_gesture_discard', g)
                else:
                    self.dispatch('on_gesture_complete', g)
        if need_cleanup:
            Clock.schedule_once(self._cleanup, timeout)

    def _cleanup(self, dt):
        """This method is scheduled from _complete_dispatcher to clean up the
        canvas and internal gesture list after a gesture is completed."""
        m = UNDERSHOOT_MARGIN
        rg = self.canvas.remove_group
        gestures = self._gestures
        for idx, g in enumerate(gestures):
            if g._cleanup_time is None:
                continue
            if g._cleanup_time <= Clock.get_time() + m:
                rg(g.id)
                del gestures[idx]
                self.dispatch('on_gesture_cleanup', g)

    def on_gesture_start(self, *l):
        pass

    def on_gesture_extend(self, *l):
        pass

    def on_gesture_merge(self, *l):
        pass

    def on_gesture_complete(self, *l):
        pass

    def on_gesture_discard(self, *l):
        pass

    def on_gesture_cleanup(self, *l):
        pass