import pytest
from kivy.tests.common import GraphicUnitTest
from kivy.base import EventLoop
from kivy.uix.bubble import Bubble
from kivy.uix.bubble import BubbleContent
from kivy.uix.bubble import BubbleButton
class _TestBubble(Bubble):

    @property
    def arrow_length(self):
        return self._arrow_image.height

    @property
    def arrow_width(self):
        return self._arrow_image.width

    @property
    def arrow_rotation(self):
        return self._arrow_image_scatter.rotation

    @property
    def arrow_layout_pos(self):
        return self._arrow_image_layout.pos

    @property
    def arrow_layout_size(self):
        return self._arrow_image_layout.size

    @property
    def arrow_center_pos_within_arrow_layout(self):
        x = self._arrow_image_scatter_wrapper.center_x
        y = self._arrow_image_scatter_wrapper.center_y
        return (x, y)