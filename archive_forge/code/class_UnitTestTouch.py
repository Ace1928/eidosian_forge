import unittest
import logging
import pytest
import sys
from functools import partial
import os
import threading
from kivy.graphics.cgl import cgl_get_backend_name
from kivy.input.motionevent import MotionEvent
class UnitTestTouch(MotionEvent):
    """Custom MotionEvent representing a single touch. Similar to `on_touch_*`
    methods from the Widget class, this one introduces:

    * touch_down
    * touch_move
    * touch_up

    Create a new touch with::

        touch = UnitTestTouch(x, y)

    then you press it on the default position with::

        touch.touch_down()

    or move it or even release with these simple calls::

        touch.touch_move(new_x, new_y)
        touch.touch_up()
    """

    def __init__(self, x, y):
        """Create a MotionEvent instance with X and Y of the first
        position a touch is at.
        """
        from kivy.base import EventLoop
        self.eventloop = EventLoop
        win = EventLoop.window
        super(UnitTestTouch, self).__init__(self.__class__.__name__, 99, {'x': x / (win.width - 1.0), 'y': y / (win.height - 1.0)}, is_touch=True, type_id='touch')
        self.profile = ['pos']

    def touch_down(self, *args):
        self.eventloop.post_dispatch_input('begin', self)

    def touch_move(self, x, y):
        win = self.eventloop.window
        self.move({'x': x / (win.width - 1.0), 'y': y / (win.height - 1.0)})
        self.eventloop.post_dispatch_input('update', self)

    def touch_up(self, *args):
        self.eventloop.post_dispatch_input('end', self)

    def depack(self, args):
        self.sx = args['x']
        self.sy = args['y']
        super().depack(args)