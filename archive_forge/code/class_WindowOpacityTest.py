from itertools import product
from kivy.tests import GraphicUnitTest
from kivy.logger import LoggerHistory
class WindowOpacityTest(GraphicUnitTest):

    def setUp(self):
        super().setUp()
        self._prev_window_opacity = self.Window.opacity
        self._prev_history = LoggerHistory.history[:]

    def tearDown(self):
        self.Window.opacity = self._prev_window_opacity
        LoggerHistory.history[:] = self._prev_history
        super().tearDown()

    def get_new_opacity_value(self):
        opacity = self.Window.opacity
        opacity = opacity - 0.1 if opacity >= 0.9 else opacity + 0.1
        return round(opacity, 2)

    def check_opacity_support(self):
        LoggerHistory.clear_history()
        self.Window.opacity = self.get_new_opacity_value()
        return not LoggerHistory.history

    def test_window_opacity_property(self):
        if self.check_opacity_support():
            opacity = self.get_new_opacity_value()
            self.Window.opacity = opacity
            self.assertEqual(self.Window.opacity, opacity)

    def test_window_opacity_clamping_positive(self):
        if self.check_opacity_support():
            self.Window.opacity = 1.5
            self.assertEqual(self.Window.opacity, 1.0)

    def test_window_opacity_clamping_negative(self):
        if self.check_opacity_support():
            self.Window.opacity = -1.5
            self.assertEqual(self.Window.opacity, 0.0)