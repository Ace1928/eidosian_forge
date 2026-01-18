import sys
import enum
import warnings
import operator
from pyglet.event import EventDispatcher
def add_button(control):
    setattr(self, control.name + '_control', control)

    @control.event
    def on_press():
        self.dispatch_event('on_button_press', control.name)

    @control.event
    def on_release():
        self.dispatch_event('on_button_release', control.name)