import torch
from torch._C import _rename_privateuse1_backend, _get_privateuse1_backend_name
from typing import List, Optional, Union
def _generate_storage_methods_for_privateuse1_backend(custom_backend_name: str, unsupported_dtype: Optional[List[torch.dtype]]=None) -> None:

    @property
    def wrap_storage_backend(self: torch.storage._StorageBase) -> bool:
        """Return the internal :class:`torch.UntypedStorage`."""
        return self.device.type == custom_backend_name
    _check_register_once(torch.storage._StorageBase, f'is_{custom_backend_name}')
    setattr(torch.storage._StorageBase, f'is_{custom_backend_name}', wrap_storage_backend)

    def wrap_storage_to(self, device=None, non_blocking=False):
        """Return a copy of this object in custom device memory.

        If this object is already in device memory and on the correct device, then
        no copy is performed and the original object is returned.

        Args:
            device (int): The destination device id. Defaults to the current device.
            non_blocking (bool): If ``True`` and the source is in pinned memory,
            the copy will be asynchronous with respect to the host. Otherwise,
            the argument has no effect.
        """
        device_idx = _normalization_device(custom_backend_name, device)
        if getattr(self, f'is_{custom_backend_name}'):
            if self.get_device() == device_idx:
                return self
        if self.is_sparse:
            raise RuntimeError(f'Can not support a sparse storage move to {custom_backend_name} backend')
        untyped_storage = torch.UntypedStorage(self.size(), device=torch.device(f'{custom_backend_name}:{device_idx}'))
        untyped_storage.copy_(self, non_blocking)
        return untyped_storage
    _check_register_once(torch.storage._StorageBase, custom_backend_name)
    setattr(torch.storage._StorageBase, custom_backend_name, wrap_storage_to)

    @property
    def wrap_typed_storage_backend(self: torch.storage.TypedStorage) -> bool:
        torch.storage._warn_typed_storage_removal()
        return self._untyped_storage.device.type == custom_backend_name
    _check_register_once(torch.TypedStorage, f'is_{custom_backend_name}')
    setattr(torch.storage.TypedStorage, f'is_{custom_backend_name}', wrap_typed_storage_backend)

    def wrap_typed_storage_to(self: torch.storage.TypedStorage, device=None, non_blocking=False, **kwargs) -> torch.storage.TypedStorage:
        torch.storage._warn_typed_storage_removal()
        if unsupported_dtype and self.dtype in unsupported_dtype:
            raise RuntimeError(f'Cannot create {custom_backend_name} storage as {self.dtype} dtype is not supported by this backend')
        custom_backend_storage: torch.UntypedStorage = getattr(self._untyped_storage, custom_backend_name)(device, non_blocking, **kwargs)
        return self._new_wrapped_storage(custom_backend_storage)
    _check_register_once(torch.TypedStorage, custom_backend_name)
    setattr(torch.TypedStorage, custom_backend_name, wrap_typed_storage_to)