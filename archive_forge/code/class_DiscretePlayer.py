from __future__ import annotations
from typing import (
import param
from ..config import config
from ..models.widgets import Player as _BkPlayer
from ..util import indexOf, isIn
from .base import Widget
from .select import SelectBase
class DiscretePlayer(PlayerBase, SelectBase):
    """
    The `DiscretePlayer` provides controls to iterate through a list of
    discrete options.  The speed at which the widget plays is defined
    by the `interval` (in milliseconds), but it is also possible to skip items using the
    `step` parameter.

    Reference: https://panel.holoviz.org/reference/widgets/DiscretePlayer.html

    :Example:

    >>> DiscretePlayer(
    ...     name='Discrete Player',
    ...     options=[2, 4, 8, 16, 32, 64, 128], value=32,
    ...     loop_policy='loop'
    ... )
    """
    interval = param.Integer(default=500, doc='Interval between updates')
    value = param.Parameter(doc='Current player value')
    value_throttled = param.Parameter(constant=True, doc='Current player value')
    _rename: ClassVar[Mapping[str, str | None]] = {'name': None, 'options': None}
    _source_transforms: ClassVar[Mapping[str, str | None]] = {'value': None, 'value_throttled': None}

    def _process_param_change(self, msg):
        values = self.values
        if 'options' in msg:
            msg['start'] = 0
            msg['end'] = len(values) - 1
            if values and (not isIn(self.value, values)):
                self.value = values[0]
        if 'value' in msg:
            value = msg['value']
            if isIn(value, values):
                msg['value'] = indexOf(value, values)
            elif values:
                self.value = values[0]
        if 'value_throttled' in msg:
            del msg['value_throttled']
        return super()._process_param_change(msg)

    def _process_property_change(self, msg):
        for prop in ('value', 'value_throttled'):
            if prop in msg:
                value = msg.pop(prop)
                if value < len(self.options):
                    msg[prop] = self.values[value]
        return msg