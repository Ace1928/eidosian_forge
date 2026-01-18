import sys
import enum
import warnings
import operator
from pyglet.event import EventDispatcher
def _bind_axis_control(self, relation, control, axis_name):
    if not (control.min or control.max):
        warnings.warn(f"Control('{control.name}') min & max values are both 0. Skipping.")
        return
    tscale = 1.0 / (control.max - control.min)
    scale = 2.0 / (control.max - control.min)
    bias = -1.0 - control.min * scale
    if control.inverted:
        scale = -scale
        bias = -bias
    if relation.sign in (Sign.POSITIVE, Sign.NEGATIVE):
        setattr(self, f'_{axis_name}_sign', relation.sign)
    dpad_comparison_map = {Sign.NEGATIVE: (operator.lt, -0.1), Sign.POSITIVE: (operator.gt, 0.1)}
    if axis_name in ('dpup', 'dpdown'):

        @control.event
        def on_change(value):
            normalized_value = value * scale + bias
            compare, limit = dpad_comparison_map[self._dpup_sign]
            setattr(self, 'dpup', compare(normalized_value, limit))
            compare, limit = dpad_comparison_map[self._dpdown_sign]
            setattr(self, 'dpdown', compare(normalized_value, limit))
            self.dispatch_event('on_dpad_motion', self, self.dpleft, self.dpright, self.dpup, self.dpdown)
    elif axis_name in ('dpleft', 'dpright'):

        @control.event
        def on_change(value):
            normalized_value = value * scale + bias
            compare, limit = dpad_comparison_map[self._dpleft_sign]
            setattr(self, 'dpleft', compare(normalized_value, limit))
            compare, limit = dpad_comparison_map[self._dpright_sign]
            setattr(self, 'dpright', compare(normalized_value, limit))
            self.dispatch_event('on_dpad_motion', self, self.dpleft, self.dpright, self.dpup, self.dpdown)
    elif axis_name in ('lefttrigger', 'righttrigger'):

        @control.event
        def on_change(value):
            normalized_value = value * tscale
            setattr(self, axis_name, normalized_value)
            self.dispatch_event('on_trigger_motion', self, axis_name, normalized_value)
    elif axis_name in ('leftx', 'lefty'):

        @control.event
        def on_change(value):
            setattr(self, axis_name, value * scale + bias)
            self.dispatch_event('on_stick_motion', self, 'leftstick', self.leftx, -self.lefty)
    elif axis_name in ('rightx', 'righty'):

        @control.event
        def on_change(value):
            setattr(self, axis_name, value * scale + bias)
            self.dispatch_event('on_stick_motion', self, 'rightstick', self.rightx, -self.righty)