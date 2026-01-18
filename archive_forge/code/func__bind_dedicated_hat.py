import sys
import enum
import warnings
import operator
from pyglet.event import EventDispatcher
def _bind_dedicated_hat(self, relation, control):

    @control.event
    def on_change(value):
        if value & 65535 == 65535:
            self.dpleft = self.dpright = self.dpup = self.dpdown = False
        else:
            if control.max > 8:
                value //= 4095
            if 0 <= value < 8:
                self.dpleft, self.dpright, self.dpup, self.dpdown = ((False, False, True, False), (False, True, True, False), (False, True, False, False), (False, True, False, True), (False, False, False, True), (True, False, False, True), (True, False, False, False), (True, False, True, False))[value]
            else:
                self.dpleft = self.dpright = self.dpup = self.dpdown = False
        self.dispatch_event('on_dpad_motion', self, self.dpleft, self.dpright, self.dpup, self.dpdown)