import collections
import json
from numbers import Complex
from typing import (
import numpy as np
from deprecated import deprecated
from deprecated.sphinx import versionadded
from pyquil.quilatom import (
from dataclasses import dataclass
@dataclass
class DefFrame(AbstractInstruction):
    frame: Frame
    ' The frame being defined. '
    direction: Optional[str] = None
    " The direction of the frame, i.e. 'tx' or 'rx'. "
    initial_frequency: Optional[float] = None
    ' The initial frequency of the frame. '
    hardware_object: Optional[str] = None
    ' The name of the hardware object associated to the frame. '
    sample_rate: Optional[float] = None
    ' The sample rate of the frame [Hz]. '
    center_frequency: Optional[float] = None
    " The 'center' frequency of the frame, used for detuning arithmetic. "
    enable_raw_capture: Optional[str] = None
    ' A flag signalling that raw capture is enabled for this frame. '
    channel_delay: Optional[float] = None
    ' The amount to delay this frame, to account for hardware signal offsets [seconds]. '

    def out(self) -> str:
        r = f'DEFFRAME {self.frame.out()}'
        options = [(self.direction, 'DIRECTION'), (self.initial_frequency, 'INITIAL-FREQUENCY'), (self.center_frequency, 'CENTER-FREQUENCY'), (self.hardware_object, 'HARDWARE-OBJECT'), (self.sample_rate, 'SAMPLE-RATE'), (self.enable_raw_capture, 'ENABLE-RAW-CAPTURE'), (self.channel_delay, 'CHANNEL-DELAY')]
        if any((value for value, name in options)):
            r += ':'
            for value, name in options:
                if value is None:
                    continue
                else:
                    r += f'\n    {name}: {json.dumps(value)}'
        return r + '\n'