import sys
import pytest
import itertools
from threading import Thread
from kivy.tests.common import GraphicUnitTest, requires_graphics
class SmoothVertexInstructionTest(GraphicUnitTest):

    def _convert_points(self, points):
        if points and isinstance(points[0], (list, tuple)):
            return list(itertools.chain(*points))
        else:
            return list(points)

    def _filtered_points(self, points):
        index = 0
        p = self._convert_points(points)
        if len(p) < 6:
            return []
        while index < len(p) - 2:
            x1, y1 = (p[index], p[index + 1])
            x2, y2 = (p[index + 2], p[index + 3])
            if abs(x2 - x1) < 1.0 and abs(y2 - y1) < 1.0:
                del p[index + 2:index + 4]
            else:
                index += 2
        if abs(p[0] - p[-2]) < 1.0 and abs(p[1] - p[-1]) < 1.0:
            del p[:2]
        return p

    def _get_texture(self):
        from kivy.graphics.texture import Texture
        return Texture.create()

    def test_antialiasing_line(self):
        from kivy.uix.widget import Widget
        from kivy.graphics import Color, Rectangle, Instruction
        from kivy.graphics.vertex_instructions import AntiAliasingLine
        r = self.render
        with pytest.raises(TypeError):
            AntiAliasingLine(None, points=[10, 20, 30, 20, 30, 10])
        target_rect = Rectangle()
        AntiAliasingLine(target_rect, points=[10, 20, 30, 40, 50, 60])
        pixels = b'\xff\xff\xff\x00\xff\xff\xff\xff\xff\xff\xff\x00'
        instruction = Instruction()
        aa_line = AntiAliasingLine(instruction)
        assert aa_line.texture.pixels == pixels
        assert aa_line.width == 2.5
        points_1 = [51.0, 649.0, 199.0, 649.0, 199.0, 501.0, 51.0, 501.0]
        points_2 = [261.0, 275.0, 335.0, 349.0, 335.0, 349.0, 409.0, 275.0, 409.0, 275.0, 335.0, 201.0, 335.0, 201.0, 261.0, 275.0]
        points_3 = [260.0, 275.0, 261.0, 275.0, 261.0, 275.0, 261.999999999999, 275.99999999, 261.06667650085353, 278.14064903651496, 261.26658584785304, 281.2756384111877, 261.56658584785305, 281.3756384111877, 261.5993677908431, 284.39931866126904, 262.0644226342696, 287.50606070381684, 262.0644226342696, 287.50606070381684, 262.6609123178712, 290.59026597968375, 263.3877619269211, 293.6463765424993, 264.2436616292954, 296.66888507446475, 265.22706903587977, 299.65234481091227, 265.22706903587977, 299.65234481091227, 266.3362119800583, 302.59137935574284, 267.5690917112779, 305.48069237005546, 268.9234864969319, 308.31507711650784, 270.39695562607204, 311.089425842209, 270.89695562607204, 311.589425842209, 271.98684380773494, 313.7987389832352, 273.69028595595563, 316.4381341741821, 275.50421235284637, 319.00285504651725, 275.50421235284637, 319.00285504651725, 277.4253541804354, 321.48827979987755, 279.45024941129833, 323.8899295308661, 281.57524904736516, 326.20347630433844, 283.79652369566156, 328.4247509526349, 283.99652369566155, 328.7247509526349, 286.1100704691339, 330.54975058870167, 288.5117202001224, 332.5746458195646, 288.5117202001224, 332.5746458195646, 290.99714495348275, 334.4957876471537, 293.5618658258179, 336.3097140440444, 293.5618658258179, 336.3097140440444, 293.2618658258179, 336.1097140440444]
        points_4 = [100, 100, 200, 100]
        for points in (points_1, points_2, points_3, points_4):
            wid = Widget()
            with wid.canvas:
                Color(1, 1, 1, 0.5)
                inst = Instruction()
                aa_line = AntiAliasingLine(inst, points=points)
            r(wid)
            filtered_points = self._filtered_points(points)
            assert aa_line.points == filtered_points + filtered_points[:2]
        for points in (points_1, points_2, points_3, points_4):
            wid = Widget()
            with wid.canvas:
                Color(1, 1, 1, 0.5)
                inst = Instruction()
                aa_line = AntiAliasingLine(inst, points=points, close=0)
            r(wid)
            assert aa_line.points == self._filtered_points(points)

    def test_smoothrectangle(self):
        from kivy.uix.widget import Widget
        from kivy.graphics import Color, SmoothRectangle
        r = self.render
        wid = Widget()
        with wid.canvas:
            Color(1, 1, 1, 0.5)
            rect = SmoothRectangle(pos=(100, 100), size=(150, 150))
        r(wid)
        filtered_points = self._filtered_points(rect.points)
        assert rect.antialiasing_line_points == filtered_points + filtered_points[:2]
        rect.size = (150, -2)
        r(wid)
        assert rect.antialiasing_line_points == []
        rect.size = (150, 2)
        r(wid)
        assert rect.antialiasing_line_points == []
        rect.size = (150, 150)
        r(wid)
        assert rect.antialiasing_line_points == filtered_points + filtered_points[:2]
        rect.texture = self._get_texture()
        r(wid)
        assert rect.antialiasing_line_points == []
        rect.source = ''
        r(wid)
        assert rect.antialiasing_line_points == filtered_points + filtered_points[:2]
        wid = Widget()
        with wid.canvas:
            Color(1, 1, 1, 0.5)
            rect = SmoothRectangle(pos=(100, 100), size=(150, -3))
        r(wid)
        assert rect.antialiasing_line_points == []
        wid = Widget()
        with wid.canvas:
            Color(1, 1, 1, 0.5)
            rect = SmoothRectangle(pos=(100, 100), size=(3.99, 3.99))
        r(wid)
        assert rect.antialiasing_line_points == []

    def test_smoothroundedrectangle(self):
        from kivy.uix.widget import Widget
        from kivy.graphics import Color, SmoothRoundedRectangle
        r = self.render
        wid = Widget()
        with wid.canvas:
            Color(1, 1, 1, 0.5)
            rounded_rect = SmoothRoundedRectangle(pos=(100, 100), size=(150, 150), radius=[(10, 50), (100, 50), (0, 150), (200, 50)], segments=60)
        r(wid)
        filtered_points = self._filtered_points(rounded_rect.points)
        assert rounded_rect.antialiasing_line_points == filtered_points + filtered_points[:2]
        rounded_rect.size = (150, -2)
        r(wid)
        assert rounded_rect.antialiasing_line_points == []
        rounded_rect.size = (150, 2)
        r(wid)
        assert rounded_rect.antialiasing_line_points == []
        rounded_rect.size = (150, 150)
        r(wid)
        assert rounded_rect.antialiasing_line_points == filtered_points + filtered_points[:2]
        rounded_rect.texture = self._get_texture()
        r(wid)
        assert rounded_rect.antialiasing_line_points == []
        rounded_rect.source = ''
        r(wid)
        assert rounded_rect.antialiasing_line_points == filtered_points + filtered_points[:2]
        wid = Widget()
        with wid.canvas:
            Color(1, 1, 1, 0.5)
            rounded_rect = SmoothRoundedRectangle(pos=(100, 100), size=(150, 150), segments=0)
        r(wid)
        filtered_points = self._filtered_points(rounded_rect.points)
        assert rounded_rect.antialiasing_line_points == filtered_points + filtered_points[:2]
        wid = Widget()
        with wid.canvas:
            Color(1, 1, 1, 0.5)
            rounded_rect = SmoothRoundedRectangle(pos=(100, 100), size=(150, -3))
        r(wid)
        assert rounded_rect.antialiasing_line_points == []
        wid = Widget()
        with wid.canvas:
            Color(1, 1, 1, 0.5)
            rounded_rect = SmoothRoundedRectangle(pos=(100, 100), size=(3.99, 3.99))
        r(wid)
        assert rounded_rect.antialiasing_line_points == []

    def test_smoothellipse(self):
        from kivy.uix.widget import Widget
        from kivy.graphics import Color, SmoothEllipse
        r = self.render
        wid = Widget()
        with wid.canvas:
            Color(1, 1, 1, 0.5)
            ellipse = SmoothEllipse(pos=(100, 100), size=(150, 150))
        r(wid)
        ellipse_center = [ellipse.pos[0] + ellipse.size[0] / 2, ellipse.pos[1] + ellipse.size[1] / 2]
        filtered_points = self._filtered_points(ellipse.points + ellipse_center)
        assert ellipse.antialiasing_line_points == filtered_points + filtered_points[:2]
        ellipse.size = (150, -2)
        r(wid)
        assert ellipse.antialiasing_line_points == []
        ellipse.size = (150, 2)
        r(wid)
        assert ellipse.antialiasing_line_points == []
        ellipse.size = (150, 150)
        r(wid)
        assert ellipse.antialiasing_line_points == filtered_points + filtered_points[:2]
        ellipse.texture = self._get_texture()
        r(wid)
        assert ellipse.antialiasing_line_points == []
        ellipse.source = ''
        r(wid)
        assert ellipse.antialiasing_line_points == filtered_points + filtered_points[:2]
        wid = Widget()
        with wid.canvas:
            Color(1, 1, 1, 0.5)
            ellipse = SmoothEllipse(pos=(100, 100), size=(150, 150), angle_start=90, angle_end=-120)
        r(wid)
        ellipse_center = [ellipse.pos[0] + ellipse.size[0] / 2, ellipse.pos[1] + ellipse.size[1] / 2]
        filtered_points = self._filtered_points(ellipse.points + ellipse_center)
        assert ellipse.antialiasing_line_points == filtered_points + filtered_points[:2]
        wid = Widget()
        with wid.canvas:
            Color(1, 1, 1, 0.5)
            ellipse = SmoothEllipse(pos=(100, 100), size=(150, -3))
        r(wid)
        assert ellipse.antialiasing_line_points == []
        wid = Widget()
        with wid.canvas:
            Color(1, 1, 1, 0.5)
            ellipse = SmoothEllipse(pos=(100, 100), size=(3.99, 3.99))
        r(wid)
        assert ellipse.antialiasing_line_points == []

    def test_smoothtriangle(self):
        from kivy.uix.widget import Widget
        from kivy.graphics import Color, SmoothTriangle
        r = self.render
        wid = Widget()
        with wid.canvas:
            Color(1, 0, 0, 0.5)
            triangle = SmoothTriangle(points=[100, 100, 200, 100, 150, 200, 500, 500, 400, 400])
        r(wid)
        filtered_points = self._filtered_points(triangle.points[:6])
        assert triangle.antialiasing_line_points == filtered_points + filtered_points[:2]
        wid = Widget()
        with wid.canvas:
            Color(0, 0, 1, 0.5)
            triangle = SmoothTriangle(points=[125, 200, 200, 100, 100, 100, 500, 500, 400, 400])
        r(wid)
        filtered_points = self._filtered_points(triangle.points[:6])
        assert triangle.antialiasing_line_points == filtered_points + filtered_points[:2]
        wid = Widget()
        with wid.canvas:
            Color(0, 1, 0, 0.5)
            triangle = SmoothTriangle(points=[100, 100, 100.5, 100, 100, 100.5])
        r(wid)
        assert triangle.antialiasing_line_points == []
        triangle.points = [125, 200, 200, 100, 100, 100]
        r(wid)
        assert triangle.antialiasing_line_points == filtered_points + filtered_points[:2]
        triangle.texture = self._get_texture()
        r(wid)
        assert triangle.antialiasing_line_points == []
        triangle.source = ''
        r(wid)
        assert triangle.antialiasing_line_points == filtered_points + filtered_points[:2]

    def test_smoothquad(self):
        from kivy.uix.widget import Widget
        from kivy.graphics import Color, SmoothQuad
        r = self.render
        wid = Widget()
        with wid.canvas:
            Color(1, 0, 0, 0.5)
            quad = SmoothQuad(points=[100, 100, 100, 200, 200, 200, 200, 100])
        r(wid)
        filtered_points = self._filtered_points(quad.points)
        assert quad.antialiasing_line_points == filtered_points + filtered_points[:2]
        wid = Widget()
        with wid.canvas:
            Color(1, 0, 0, 0.5)
            quad = SmoothQuad(points=[200, 100, 200, 200, 100, 200, 100, 100])
        r(wid)
        filtered_points = self._filtered_points(quad.points)
        assert quad.antialiasing_line_points == filtered_points + filtered_points[:2]
        wid = Widget()
        with wid.canvas:
            Color(0, 1, 0, 0.5)
            quad = SmoothQuad(points=[200, 100, 200, 100.8, 100, 100.8, 100, 100])
        r(wid)
        assert quad.antialiasing_line_points == []
        quad.points = [200, 100, 200, 200, 100, 200, 100, 100]
        r(wid)
        assert quad.antialiasing_line_points == filtered_points + filtered_points[:2]
        quad.texture = self._get_texture()
        r(wid)
        assert quad.antialiasing_line_points == []
        quad.source = ''
        r(wid)
        assert quad.antialiasing_line_points == filtered_points + filtered_points[:2]