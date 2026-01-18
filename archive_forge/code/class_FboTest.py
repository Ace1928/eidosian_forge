import unittest
from kivy.tests.common import GraphicUnitTest
from kivy.uix.widget import Widget
from kivy.graphics import Fbo, Color, Rectangle
class FboTest(Widget):

    def __init__(self, **kwargs):
        super(FboTest, self).__init__(**kwargs)
        self.positions = [(260.0, 260.0), (192.0, 192.0), (96.0, 192.0), (192.0, 96.0), (96.0, 96.0), (32.0, 192.0), (192.0, 32.0), (32.0, 32.0)]
        self.fbo = Fbo(size=(256, 256))
        with self.fbo:
            Color(0.56789, 0, 0, 1)
            Rectangle(size=(256, 64))
            Color(0, 0.56789, 0, 1)
            Rectangle(size=(64, 256))
            Color(0.56789, 0, 0, 0.5)
            Rectangle(pos=(64, 64), size=(192, 64))
            Color(0, 0.56789, 0, 0.5)
            Rectangle(pos=(64, 64), size=(64, 192))
        self.fbo.draw()