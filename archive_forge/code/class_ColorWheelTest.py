from kivy.tests.common import GraphicUnitTest, UTMotionEvent
from kivy.uix.colorpicker import ColorWheel, ColorPicker
class ColorWheelTest(GraphicUnitTest):

    def test_render(self):
        color_wheel = ColorWheel()
        self.render(color_wheel)

    def test_clicks(self):
        color_wheel = ColorWheel()
        self.assertEqual(color_wheel.color, [0, 0, 0, 0])
        pos = (color_wheel.pos[0], color_wheel.pos[1])
        touch = UTMotionEvent('unittest', 1, {'x': pos[0], 'y': pos[1]})
        touch.grab_current = color_wheel
        touch.pos = pos
        color_wheel.on_touch_down(touch)
        color_wheel.on_touch_up(touch)
        self.assertEqual(color_wheel.color, [0, 0, 0, 0])
        pos = (color_wheel.pos[0] + color_wheel.size[0] / 2, color_wheel.pos[1] + color_wheel.size[1] / 4)
        touch = UTMotionEvent('unittest', 1, {'x': pos[0], 'y': pos[1]})
        touch.grab_current = color_wheel
        touch.pos = pos
        color_wheel.on_touch_down(touch)
        color_wheel.on_touch_up(touch)
        self.assertEqual(color_wheel.color, [0.75, 0.5, 1, 1])