import inspect
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Type, Union
import torch
from torch._streambase import _EventBase, _StreamBase
class CudaInterface(DeviceInterface):
    device = torch.cuda.device
    Event = torch.cuda.Event
    Stream = torch.cuda.Stream

    class Worker:

        @staticmethod
        def set_device(device: int):
            caching_worker_current_devices['cuda'] = device

        @staticmethod
        def current_device() -> int:
            if 'cuda' in caching_worker_current_devices:
                return caching_worker_current_devices['cuda']
            return torch.cuda.current_device()

        @staticmethod
        def get_device_properties(device: _device_t=None):
            if device is not None:
                if isinstance(device, str):
                    device = torch.device(device)
                    assert device.type == 'cuda'
                if isinstance(device, torch.device):
                    device = device.index
            if device is None:
                device = CudaInterface.Worker.current_device()
            if 'cuda' not in caching_worker_device_properties:
                device_prop = [torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]
                caching_worker_device_properties['cuda'] = device_prop
            return caching_worker_device_properties['cuda'][device]
    current_device = staticmethod(torch.cuda.current_device)
    set_device = staticmethod(torch.cuda.set_device)
    device_count = staticmethod(torch.cuda.device_count)
    stream = staticmethod(torch.cuda.stream)
    current_stream = staticmethod(torch.cuda.current_stream)
    set_stream = staticmethod(torch.cuda.set_stream)
    _set_stream_by_id = staticmethod(torch.cuda._set_stream_by_id)
    synchronize = staticmethod(torch.cuda.synchronize)
    get_device_properties = staticmethod(torch.cuda.get_device_properties)
    get_raw_stream = staticmethod(get_cuda_stream)

    @staticmethod
    def is_available() -> bool:
        return torch.cuda.is_available()

    @staticmethod
    def get_compute_capability(device: _device_t=None):
        major, min = torch.cuda.get_device_capability(device)
        return major * 10 + min