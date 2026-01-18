import pytest
from kivy.tests.common import GraphicUnitTest
from kivy.base import EventLoop
from kivy.uix.bubble import Bubble
from kivy.uix.bubble import BubbleContent
from kivy.uix.bubble import BubbleButton
class _TestBubbleContent(BubbleContent):

    def update_size(self, instance, value):
        self.size = self.minimum_size

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bind(minimum_size=self.update_size)