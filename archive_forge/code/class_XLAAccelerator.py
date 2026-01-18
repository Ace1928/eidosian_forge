import functools
from typing import Any, List, Union
import torch
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import override
from lightning_fabric.accelerators.accelerator import Accelerator
from lightning_fabric.accelerators.registry import _AcceleratorRegistry
from lightning_fabric.utilities.device_parser import _check_data_type
class XLAAccelerator(Accelerator):
    """Accelerator for XLA devices, normally TPUs.

    .. warning::  Use of this accelerator beyond import and instantiation is experimental.

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if not _XLA_AVAILABLE:
            raise ModuleNotFoundError(str(_XLA_AVAILABLE))
        if not _using_pjrt():
            raise RuntimeError('The XLA XRT runtime is not supported anymore.')
        super().__init__(*args, **kwargs)

    @override
    def setup_device(self, device: torch.device) -> None:
        pass

    @override
    def teardown(self) -> None:
        pass

    @staticmethod
    @override
    def parse_devices(devices: Union[int, str, List[int]]) -> Union[int, List[int]]:
        """Accelerator device parsing logic."""
        return _parse_tpu_devices(devices)

    @staticmethod
    @override
    def get_parallel_devices(devices: Union[int, List[int]]) -> List[torch.device]:
        """Gets parallel devices for the Accelerator."""
        devices = _parse_tpu_devices(devices)
        if isinstance(devices, int):
            return [torch.device('xla', i) for i in range(devices)]
        return [torch.device('xla', devices[0])]

    @staticmethod
    @override
    @functools.lru_cache(maxsize=1)
    def auto_device_count() -> int:
        """Get the devices when set to auto."""
        if not _XLA_AVAILABLE:
            return 0
        if _XLA_GREATER_EQUAL_2_1:
            from torch_xla._internal import tpu
            return tpu.num_available_devices()
        from torch_xla.experimental import tpu
        device_count_on_version = {2: 8, 3: 8, 4: 4}
        return device_count_on_version.get(tpu.version(), 8)

    @staticmethod
    @override
    @functools.lru_cache(maxsize=1)
    def is_available() -> bool:
        try:
            return XLAAccelerator.auto_device_count() > 0
        except (ValueError, AssertionError, OSError):
            return False

    @classmethod
    @override
    def register_accelerators(cls, accelerator_registry: _AcceleratorRegistry) -> None:
        accelerator_registry.register('tpu', cls, description=cls.__name__)