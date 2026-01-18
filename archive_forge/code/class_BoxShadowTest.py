import sys
import pytest
import itertools
from threading import Thread
from kivy.tests.common import GraphicUnitTest, requires_graphics
class BoxShadowTest(GraphicUnitTest):

    def test_create(self):
        from kivy.graphics.boxshadow import BoxShadow
        from kivy.uix.widget import Widget
        from kivy.graphics import Color
        r = self.render
        wid = Widget()
        with wid.canvas:
            Color(1, 0, 0, 1)
            bs = BoxShadow(pos=(50, 50), size=(150, 150), offset=(0, 10), spread_radius=(10, -10), border_radius=(10, 10, 10, 10), blur_radius=80)
        r(wid)
        wid = Widget()
        with wid.canvas:
            Color(0, 1, 0, 1)
            bs = BoxShadow(inset=True, pos=(50, 50), size=(150, 150), offset=(0, 10), spread_radius=(10, -10), border_radius=(10, 10, 10, 10), blur_radius=80)
        r(wid)
        wid = Widget()
        with wid.canvas:
            Color(0, 0, 1, 1)
            bs = BoxShadow()
        bs.inset = True
        bs.pos = [50, 50]
        bs.size = [150, 150]
        bs.offset = [0, 10]
        bs.spread_radius = [10, -10]
        bs.border_radius = [10, 10, 10, 10]
        bs.blur_radius = 40
        r(wid)

    def test_adjusted_size(self):
        from kivy.graphics.boxshadow import BoxShadow
        raw_size = (150, 150)
        bs = BoxShadow()
        bs.pos = (50, 50)
        bs.size = raw_size
        bs.blur_radius = 80
        bs.spread_radius = (-10, 10)
        adjusted_size = (max(0, raw_size[0] + bs.blur_radius * 3 + bs.spread_radius[0] * 2), max(0, raw_size[1] + bs.blur_radius * 3 + bs.spread_radius[1] * 2))
        assert bs.size == adjusted_size
        bs.inset = True
        assert bs.size == raw_size
        bs.inset = False
        assert bs.size == adjusted_size
        bs = BoxShadow(inset=True, pos=(50, 50), size=raw_size, blur_radius=80, spread_radius=(10, -10))
        adjusted_size = (max(0, raw_size[0] + bs.blur_radius * 3 + bs.spread_radius[0] * 2), max(0, raw_size[1] + bs.blur_radius * 3 + bs.spread_radius[1] * 2))
        assert bs.size == raw_size
        bs.inset = False
        assert bs.size == adjusted_size
        bs.inset = True
        assert bs.size == raw_size

    def test_adjusted_pos(self):
        from kivy.graphics.boxshadow import BoxShadow
        raw_pos = (50, 50)
        raw_size = (150, 150)
        offset = (10, -100)
        bs = BoxShadow()
        bs.pos = raw_pos
        bs.size = raw_size
        bs.offset = offset
        bs.blur_radius = 80
        bs.spread_radius = (-10, 10)
        adjusted_pos = (raw_pos[0] - bs.blur_radius * 1.5 - bs.spread_radius[0] + bs.offset[0], raw_pos[0] - bs.blur_radius * 1.5 - bs.spread_radius[1] + bs.offset[1])
        assert bs.pos == adjusted_pos
        bs.inset = True
        assert bs.pos == raw_pos
        bs.inset = False
        assert bs.pos == adjusted_pos
        bs = BoxShadow(inset=True, pos=raw_pos, size=raw_size, offset=offset, blur_radius=80, spread_radius=(10, -10))
        adjusted_pos = (raw_pos[0] - bs.blur_radius * 1.5 - bs.spread_radius[0] + bs.offset[0], raw_pos[0] - bs.blur_radius * 1.5 - bs.spread_radius[1] + bs.offset[1])
        assert bs.pos == raw_pos
        bs.inset = False
        assert bs.pos == adjusted_pos
        bs.inset = True
        assert bs.pos == raw_pos

    def test_bounded_properties(self):
        from kivy.graphics.boxshadow import BoxShadow
        bs = BoxShadow()
        bs.pos = (50, 50)
        bs.size = (150, 150)
        bs.offset = (10, -100)
        bs.blur_radius = -80
        bs.spread_radius = (-200, -100)
        bs.border_radius = (0, 0, 100, 0)
        assert bs.size == (0, 0)
        assert bs.blur_radius == 0
        assert bs.border_radius == tuple(map(lambda value: max(1.0, min(value, min(bs.size) / 2)), bs.border_radius))
        bs = BoxShadow(pos=(50, 50), size=(150, 150), offset=(10, -100), blur_radius=-80, spread_radius=(-200, -100), border_radius=(0, 0, 100, 0))
        assert bs.size == (0, 0)
        assert bs.blur_radius == 0
        assert bs.border_radius == tuple(map(lambda value: max(1.0, min(value, min(bs.size) / 2)), bs.border_radius))

    def test_canvas_management(self):
        from kivy.graphics.boxshadow import BoxShadow
        from kivy.uix.widget import Widget
        from kivy.graphics import Color
        r = self.render
        wid = Widget()
        with wid.canvas:
            bs = BoxShadow()
        r(wid)
        assert bs in wid.canvas.children
        wid = Widget()
        bs = BoxShadow()
        wid.canvas.add(Color(1, 0, 0, 1))
        wid.canvas.add(bs)
        r(wid)
        assert bs in wid.canvas.children
        wid.canvas.remove(bs)
        assert bs not in wid.canvas.children
        wid.canvas.insert(1, bs)
        assert bs in wid.canvas.children
        assert wid.canvas.children.index(bs) == 1