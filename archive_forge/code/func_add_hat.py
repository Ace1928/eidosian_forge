import sys
import enum
import warnings
import operator
from pyglet.event import EventDispatcher
def add_hat(control):
    self.hat_x_control = control
    self.hat_y_control = control

    @control.event
    def on_change(value):
        if value & 65535 == 65535:
            self.hat_x = self.hat_y = 0
        else:
            if control.max > 8:
                value //= 4095
            if 0 <= value < 8:
                self.hat_x, self.hat_y = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1))[value]
            else:
                self.hat_x = self.hat_y = 0
        self.dispatch_event('on_joyhat_motion', self, self.hat_x, self.hat_y)