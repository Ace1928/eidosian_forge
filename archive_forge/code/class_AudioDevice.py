from abc import ABCMeta, abstractmethod
from enum import Enum, auto
from typing import Dict, Optional
from pyglet import event
class AudioDevice:
    """Base class for a platform independent audio device.
       _platform_state and _platform_flow is used to make device state numbers."""
    platform_state: Dict[int, DeviceState] = {}
    platform_flow: Dict[int, DeviceFlow] = {}

    def __init__(self, dev_id: str, name: str, description: str, flow: int, state: int):
        self.id = dev_id
        self.flow = flow
        self.state = state
        self.name = name
        self.description = description

    def __repr__(self):
        return "{}(name='{}', state={}, flow={})".format(self.__class__.__name__, self.name, self.platform_state[self.state].name, self.platform_flow[self.flow].name)