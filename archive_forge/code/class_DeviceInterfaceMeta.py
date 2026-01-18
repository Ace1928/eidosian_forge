import inspect
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Type, Union
import torch
from torch._streambase import _EventBase, _StreamBase
class DeviceInterfaceMeta(type):

    def __new__(metacls, *args, **kwargs):
        class_member = args[2]
        if 'Event' in class_member:
            assert inspect.isclass(class_member['Event']) and issubclass(class_member['Event'], _EventBase), 'DeviceInterface member Event should be inherit from _EventBase'
        if 'Stream' in class_member:
            assert inspect.isclass(class_member['Stream']) and issubclass(class_member['Stream'], _StreamBase), 'DeviceInterface member Stream should be inherit from _StreamBase'
        return super().__new__(metacls, *args, **kwargs)