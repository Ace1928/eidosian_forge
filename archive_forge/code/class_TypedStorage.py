import io
import torch
from ._utils import _type, _cuda, _hpu
from torch.types import Storage
from typing import cast, Any, Dict as _Dict, Optional as _Optional, TypeVar, Type, Union
import copy
import collections
from functools import lru_cache
import warnings
import threading
import functools
class TypedStorage:
    is_sparse = False
    dtype: torch.dtype

    @property
    def _dtype(self):
        return self.dtype

    @property
    def filename(self) -> _Optional[str]:
        """Returns the file name associated with this storage if the storage was memory mapped from a file.
           or ``None`` if the storage was not created by memory mapping a file."""
        return self._untyped_storage.filename

    def fill_(self, value):
        _warn_typed_storage_removal()
        self._setitem(slice(0, self._size()), value)
        return self

    def __new__(cls, *args, wrap_storage=None, dtype=None, device=None, _internal=False):
        if not _internal:
            _warn_typed_storage_removal()
        if cls == torch.storage._LegacyStorage:
            raise RuntimeError('Only child classes of _LegacyStorage can be instantiated')
        if cls == TypedStorage:
            return super().__new__(cls)
        else:
            arg_error_msg = f'{cls}.__new__ received an invalid combination of arguments. Expected one of:\n * no arguments\n * (int size)\n * (Sequence data)\n * (*, UntypedStorage wrap_storage)'
            if device is not None:
                raise RuntimeError(arg_error_msg + "\nKeyword argument 'device' cannot be specified")
            if dtype is not None:
                raise RuntimeError(arg_error_msg + "\nKeyword argument 'dtype' cannot be specified")
            if wrap_storage is None:
                if len(args) > 1:
                    raise RuntimeError(arg_error_msg + '\nToo many positional arguments')
                if len(args) == 1 and (not _isint(args[0])) and (not isinstance(args[0], collections.abc.Sequence)):
                    raise TypeError(arg_error_msg + f'\nArgument type not recognized: {type(args[0])}')
                return TypedStorage(*args, dtype=cls._dtype, device=_get_device_from_module(cls.__module__), _internal=True)
            else:
                if len(args) != 0:
                    raise RuntimeError(arg_error_msg + "\nNo positional arguments should be given when using 'wrap_storage'")
                if not isinstance(wrap_storage, torch.UntypedStorage):
                    raise TypeError(arg_error_msg + f"\nArgument 'wrap_storage' must be UntypedStorage, but got {type(wrap_storage)}")
                cls_device = _get_device_from_module(cls.__module__)
                if wrap_storage.device.type != cls_device:
                    raise RuntimeError(arg_error_msg + f"\nDevice of 'wrap_storage' must be {cls_device}, but got {wrap_storage.device.type}")
                return TypedStorage(*args, wrap_storage=wrap_storage, dtype=cls.dtype, _internal=True)

    def __init__(self, *args, device=None, dtype=None, wrap_storage=None, _internal=False):
        if not _internal:
            _warn_typed_storage_removal()
        arg_error_msg = 'TypedStorage.__init__ received an invalid combination of arguments. Expected one of:\n * (*, torch.device device, torch.dtype dtype)\n * (int size, *, torch.device device, torch.dtype dtype)\n * (Sequence data, *, torch.device device, torch.dtype dtype)\n * (*, UntypedStorage wrap_storage, torch.dtype dtype)'
        if wrap_storage is not None:
            if len(args) != 0:
                raise RuntimeError(arg_error_msg + "\nNo positional arguments should be given when using 'wrap_storage'")
            if dtype is None:
                raise RuntimeError(arg_error_msg + "\nArgument 'dtype' must be specified")
            if not isinstance(dtype, torch.dtype):
                raise TypeError(arg_error_msg + f"\nArgument 'dtype' must be torch.dtype, not {type(dtype)}")
            if device is not None:
                raise RuntimeError(arg_error_msg + "\nArgument 'device' should not be specified when 'wrap_storage' is given")
            self.dtype = dtype
            if not isinstance(wrap_storage, torch.UntypedStorage):
                raise TypeError(arg_error_msg + f"\nArgument 'wrap_storage' must be UntypedStorage, but got {type(wrap_storage)}")
            self._untyped_storage = wrap_storage
        else:
            self.dtype = torch.get_default_dtype() if dtype is None else dtype
            device = torch.device('cpu' if device is None else device)
            if self.dtype in [torch.quint8, torch.quint4x2, torch.quint2x4, torch.qint32, torch.qint8]:
                if device.type == 'cuda':
                    raise RuntimeError('Cannot create CUDA storage with quantized dtype')
            if len(args) == 0:
                self._untyped_storage = torch.UntypedStorage(device=device)
            elif len(args) == 1:
                if _isint(args[0]):
                    self._untyped_storage = torch.UntypedStorage(int(args[0]) * self._element_size(), device=device)
                elif isinstance(args[0], collections.abc.Sequence):
                    self._untyped_storage = _get_storage_from_sequence(args[0], self.dtype, device)
                else:
                    raise TypeError(arg_error_msg + f'\nArgument type not recognized: {type(args[0])}')
            else:
                raise RuntimeError(arg_error_msg + '\nToo many positional arguments')

    @property
    def is_cuda(self):
        _warn_typed_storage_removal()
        return self._untyped_storage.device.type == 'cuda'

    @property
    def is_hpu(self):
        _warn_typed_storage_removal()
        return self._untyped_storage.device.type == 'hpu'

    def untyped(self):
        """Return the internal :class:`torch.UntypedStorage`."""
        _warn_typed_storage_removal()
        return self._untyped_storage

    def _new_wrapped_storage(self, untyped_storage):
        assert type(untyped_storage) == torch.UntypedStorage
        if type(self) == TypedStorage:
            return TypedStorage(wrap_storage=untyped_storage, dtype=self.dtype, _internal=True)
        else:
            return type(self)(wrap_storage=untyped_storage)

    def __len__(self):
        _warn_typed_storage_removal()
        return self._size()

    def _maybe_wrap_index(self, idx, is_stop=False):
        if idx is None:
            if is_stop:
                return self._size()
            else:
                return 0
        else:
            if type(idx) != int:
                raise TypeError(f"can't index a {type(self)} with {type(idx)}")
            if is_stop:
                if idx > self._size() or idx < -self._size():
                    raise IndexError(f'index {idx} out of range for storage of size {self.size()}')
                if idx > 0:
                    return idx
                else:
                    return idx % self._size()
            else:
                if idx >= self._size() or idx < -self._size():
                    raise IndexError(f'index {idx} out of range for storage of size {self.size()}')
                return idx % self._size()

    def __setitem__(self, idx, value):
        _warn_typed_storage_removal()
        return self._setitem(idx, value)

    def _setitem(self, idx, value):
        if not isinstance(idx, (int, slice)):
            raise RuntimeError(f"can't index a {type(self)} with {type(idx)}")
        if torch.is_storage(value):
            raise RuntimeError(f'cannot set item with value type {type(value)}')
        if self.dtype in [torch.quint8, torch.quint4x2, torch.quint2x4, torch.qint32, torch.qint8]:
            interpret_dtypes = {torch.quint8: torch.uint8, torch.quint4x2: torch.uint8, torch.quint2x4: torch.uint8, torch.qint32: torch.int32, torch.qint8: torch.int8}
            tmp_dtype = interpret_dtypes[self.dtype]
            tmp_tensor = torch.tensor([], dtype=tmp_dtype, device=self._untyped_storage.device)
            tmp_tensor.set_(TypedStorage(wrap_storage=self._untyped_storage, dtype=tmp_dtype, _internal=True))
        else:
            tmp_tensor = torch.tensor([], dtype=self.dtype, device=self._untyped_storage.device).set_(self)
        tmp_tensor[idx] = value

    def __getitem__(self, idx):
        _warn_typed_storage_removal()
        return self._getitem(idx)

    def _getitem(self, idx):
        if self._untyped_storage.device.type == 'meta':
            raise NotImplementedError("Not available for 'meta' device type")
        if isinstance(idx, slice):
            raise RuntimeError('slices are only supported in UntypedStorage.__getitem__')
        elif not isinstance(idx, int):
            raise RuntimeError(f"can't index a {type(self)} with {type(idx)}")
        if self.dtype in [torch.quint8, torch.quint4x2, torch.quint2x4, torch.qint32, torch.qint8]:
            interpret_dtypes = {torch.quint8: torch.uint8, torch.quint4x2: torch.uint8, torch.quint2x4: torch.uint8, torch.qint32: torch.int32, torch.qint8: torch.int8}
            return TypedStorage(wrap_storage=self._untyped_storage, dtype=interpret_dtypes[self.dtype], _internal=True)._getitem(idx)
        idx_wrapped = self._maybe_wrap_index(idx)
        tmp_tensor = torch.tensor([], dtype=self.dtype, device=self._untyped_storage.device).set_(self)
        return tmp_tensor[idx_wrapped].item()

    def copy_(self, source: T, non_blocking: _Optional[bool]=None):
        _warn_typed_storage_removal()
        if isinstance(source, TypedStorage):
            self._untyped_storage.copy_(source._untyped_storage, non_blocking)
        else:
            self._untyped_storage.copy_(source, non_blocking)
        return self

    def nbytes(self):
        _warn_typed_storage_removal()
        return self._nbytes()

    def _nbytes(self):
        return self._untyped_storage.nbytes()

    def type(self, dtype: _Optional[str]=None, non_blocking: bool=False) -> Union[T, str]:
        _warn_typed_storage_removal()
        if dtype is None:
            legacy_class = self._get_legacy_storage_class()
            if legacy_class is not None:
                return legacy_class.__module__ + '.' + legacy_class.__name__
            return '.'.join([self.__module__, type(self).__name__])
        else:
            return self._untyped_storage.type(dtype, non_blocking)

    def cuda(self, device=None, non_blocking=False, **kwargs) -> T:
        _warn_typed_storage_removal()
        if self.dtype in [torch.quint8, torch.quint4x2, torch.quint2x4, torch.qint32, torch.qint8]:
            raise RuntimeError('Cannot create CUDA storage with quantized dtype')
        cuda_storage: torch.UntypedStorage = self._untyped_storage.cuda(device, non_blocking, **kwargs)
        return self._new_wrapped_storage(cuda_storage)

    def hpu(self, device=None, non_blocking=False, **kwargs) -> T:
        _warn_typed_storage_removal()
        if self.dtype in [torch.quint8, torch.quint4x2, torch.quint2x4, torch.qint32, torch.qint8]:
            raise RuntimeError('Cannot create HPU storage with quantized dtype')
        hpu_storage: torch.UntypedStorage = self._untyped_storage.hpu(device, non_blocking, **kwargs)
        return self._new_wrapped_storage(hpu_storage)

    def element_size(self):
        _warn_typed_storage_removal()
        return self._element_size()

    def _element_size(self):
        return torch._utils._element_size(self.dtype)

    def get_device(self) -> int:
        _warn_typed_storage_removal()
        return self._untyped_storage.get_device()

    def __str__(self):
        _warn_typed_storage_removal()
        info_str = f'[{torch.typename(self)}(dtype={self.dtype}, device={self.device}) of size {len(self)}]'
        if self.device.type == 'meta':
            return '...\n' + info_str
        else:
            data_str = ' ' + '\n '.join((str(self[i]) for i in range(self.size())))
            return data_str + '\n' + info_str

    def __repr__(self):
        _warn_typed_storage_removal()
        return str(self)

    def __iter__(self):
        _warn_typed_storage_removal()
        return iter((self[i] for i in range(self.size())))

    def __copy__(self):
        _warn_typed_storage_removal()
        return self._new_wrapped_storage(copy.copy(self._untyped_storage))

    def __deepcopy__(self, memo):
        _warn_typed_storage_removal()
        return self._deepcopy(memo)

    def _deepcopy(self, memo):
        return self._new_wrapped_storage(copy.deepcopy(self._untyped_storage, memo))

    def __sizeof__(self):
        _warn_typed_storage_removal()
        return super().__sizeof__() + self.nbytes()

    def clone(self):
        """Return a copy of this storage."""
        _warn_typed_storage_removal()
        return self._new_wrapped_storage(self._untyped_storage.clone())

    def tolist(self):
        """Return a list containing the elements of this storage."""
        _warn_typed_storage_removal()
        return list(self)

    def cpu(self):
        """Return a CPU copy of this storage if it's not already on the CPU."""
        _warn_typed_storage_removal()
        return self._new_wrapped_storage(self._untyped_storage.cpu())

    def is_pinned(self, device: Union[str, torch.device]='cuda'):
        """Determine whether the CPU TypedStorage is already pinned on device.

        Args:
            device (str or torch.device): The device to pin memory on. Default: ``'cuda'``

        Returns:
            A boolean variable.
        """
        _warn_typed_storage_removal()
        return self._untyped_storage.is_pinned(device)

    def pin_memory(self, device: Union[str, torch.device]='cuda'):
        """Copy the CPU TypedStorage to pinned memory, if it's not already pinned.

        Args:
            device (str or torch.device): The device to pin memory on. Default: ``'cuda'``.

        Returns:
            A pinned CPU storage.
        """
        _warn_typed_storage_removal()
        return self._new_wrapped_storage(self._untyped_storage.pin_memory(device=device))

    def share_memory_(self):
        """See :meth:`torch.UntypedStorage.share_memory_`"""
        _warn_typed_storage_removal()
        return self._share_memory_()

    def _share_memory_(self):
        self._untyped_storage.share_memory_()
        return self

    def _new_shared(self, size, *, device=None):
        """Create a new storage in shared memory with the same data type."""
        if device is None:
            device = 'cpu'
        device = torch.device(device)
        untyped_storage = torch.UntypedStorage._new_shared(size * self._element_size(), device=device)
        return TypedStorage(wrap_storage=untyped_storage, dtype=self.dtype, _internal=True)

    @property
    def _cdata(self):
        return self._untyped_storage._cdata

    @property
    def device(self):
        _warn_typed_storage_removal()
        return self._untyped_storage.device

    def size(self):
        _warn_typed_storage_removal()
        return self._size()

    def _size(self):
        return self._untyped_storage.nbytes() // self._element_size()

    def pickle_storage_type(self):
        _warn_typed_storage_removal()
        return self._pickle_storage_type()

    def _pickle_storage_type(self):
        try:
            return _dtype_to_storage_type_map()[self.dtype]
        except KeyError as e:
            raise KeyError(f'dtype {self.dtype} is not recognized') from e

    def __reduce__(self):
        b = io.BytesIO()
        torch.save(self, b, _use_new_zipfile_serialization=False)
        return (_load_from_bytes, (b.getvalue(),))

    def data_ptr(self):
        _warn_typed_storage_removal()
        return self._data_ptr()

    def _data_ptr(self):
        return self._untyped_storage.data_ptr()

    def resize_(self, size):
        _warn_typed_storage_removal()
        self._resize_(size)

    def _resize_(self, size):
        self._untyped_storage.resize_(size * self._element_size())

    @classmethod
    def _free_weak_ref(cls, *args, **kwargs):
        return UntypedStorage._free_weak_ref(*args, **kwargs)

    def _weak_ref(self, *args, **kwargs):
        return self._untyped_storage._weak_ref(*args, **kwargs)

    @classmethod
    def from_buffer(cls, *args, **kwargs):
        _warn_typed_storage_removal()
        return cls._from_buffer(*args, **kwargs)

    @classmethod
    def _from_buffer(cls, *args, dtype=None, device=None, **kwargs):
        if cls == TypedStorage:
            dtype = torch.get_default_dtype() if dtype is None else dtype
            device = torch.device('cpu' if device is None else device)
            if device.type != 'cpu':
                raise RuntimeError(f'TypedStorage.from_buffer: Not available for device {device.type}')
            untyped_storage: torch.UntypedStorage = torch.UntypedStorage.from_buffer(*args, dtype=dtype, **kwargs)
        else:
            if dtype is not None or len(args) == 5:
                raise RuntimeError("from_buffer: 'dtype' can only be specified in UntypedStorage.from_buffer and TypedStorage.from_buffer")
            if device is not None:
                raise RuntimeError("from_buffer: 'device' can only be specified in UntypedStorage.from_buffer and TypedStorage.from_buffer")
            dtype = cls._dtype
            untyped_storage = torch.UntypedStorage.from_buffer(*args, dtype=dtype, **kwargs)
        return TypedStorage(wrap_storage=untyped_storage, dtype=dtype, _internal=True)

    def _to(self, dtype):
        if not isinstance(dtype, torch.dtype):
            raise TypeError(f"Argument 'dtype' must be torch.dtype, not {type(dtype)}")
        storage = torch.tensor([], dtype=self.dtype, device=self.device).set_(self).to(dtype)._typed_storage()
        if storage.data_ptr() == self.data_ptr():
            storage = storage.clone()
        return storage

    def double(self):
        """Casts this storage to double type."""
        _warn_typed_storage_removal()
        return self._to(torch.double)

    def float(self):
        """Casts this storage to float type."""
        _warn_typed_storage_removal()
        return self._to(torch.float)

    def half(self):
        """Casts this storage to half type."""
        _warn_typed_storage_removal()
        return self._to(torch.half)

    def long(self):
        """Casts this storage to long type."""
        _warn_typed_storage_removal()
        return self._to(torch.long)

    def int(self):
        """Casts this storage to int type."""
        _warn_typed_storage_removal()
        return self._to(torch.int)

    def short(self):
        """Casts this storage to short type."""
        _warn_typed_storage_removal()
        return self._to(torch.short)

    def char(self):
        """Casts this storage to char type."""
        _warn_typed_storage_removal()
        return self._to(torch.int8)

    def byte(self):
        """Casts this storage to byte type."""
        _warn_typed_storage_removal()
        return self._to(torch.uint8)

    def bool(self):
        """Casts this storage to bool type."""
        _warn_typed_storage_removal()
        return self._to(torch.bool)

    def bfloat16(self):
        """Casts this storage to bfloat16 type."""
        _warn_typed_storage_removal()
        return self._to(torch.bfloat16)

    def complex_double(self):
        """Casts this storage to complex double type."""
        _warn_typed_storage_removal()
        return self._to(torch.cdouble)

    def complex_float(self):
        """Casts this storage to complex float type."""
        _warn_typed_storage_removal()
        return self._to(torch.cfloat)

    def float8_e5m2(self):
        """Casts this storage to float8_e5m2 type"""
        _warn_typed_storage_removal()
        return self._to(torch.float8_e5m2)

    def float8_e4m3fn(self):
        """Casts this storage to float8_e4m3fn type"""
        _warn_typed_storage_removal()
        return self._to(torch.float8_e4m3fn)

    @classmethod
    def from_file(cls, filename, shared, size):
        """from_file(filename, shared=False, size=0) -> Storage

        Creates a CPU storage backed by a memory-mapped file.

        If ``shared`` is ``True``, then memory is shared between all processes.
        All changes are written to the file. If ``shared`` is ``False``, then the changes on
        the storage do not affect the file.

        ``size`` is the number of elements in the storage. If ``shared`` is ``False``,
        then the file must contain at least ``size * sizeof(Type)`` bytes
        (``Type`` is the type of storage). If ``shared`` is ``True`` the file will be created if needed.

        Args:
            filename (str): file name to map
            shared (bool): whether to share memory (whether ``MAP_SHARED`` or ``MAP_PRIVATE`` is passed to the
                            underlying `mmap(2) call <https://man7.org/linux/man-pages/man2/mmap.2.html>`_)
            size (int): number of elements in the storage
        """
        _warn_typed_storage_removal()
        if cls == TypedStorage:
            raise RuntimeError('from_file can only be called on derived classes')
        untyped_storage: UntypedStorage = UntypedStorage.from_file(filename, shared, size * torch._utils._element_size(cls.dtype))
        storage = cls(wrap_storage=untyped_storage)
        return storage

    @classmethod
    def _expired(cls, *args, **kwargs):
        return UntypedStorage._expired(*args, **kwargs)

    def _write_file(self, *args, **kwargs):
        return self._untyped_storage._write_file(*args, **kwargs)

    def _set_from_file(self, *args, **kwargs):
        return self._untyped_storage._set_from_file(*args, **kwargs)

    def _set_cdata(self, *args, **kwargs):
        return self._untyped_storage._set_cdata(*args, **kwargs)

    def _share_cuda_(self, *args, **kwargs):
        return self._untyped_storage._share_cuda_(*args, **kwargs)

    def is_shared(self):
        _warn_typed_storage_removal()
        return self._is_shared()

    def _is_shared(self):
        return self._untyped_storage.is_shared()

    @classmethod
    def _new_shared_cuda(cls, *args, **kwargs):
        return torch.UntypedStorage._new_shared_cuda(*args, **kwargs)

    def _share_filename_cpu_(self, *args, **kwargs):
        manager_handle, storage_handle, size = self._untyped_storage._share_filename_cpu_(*args, **kwargs)
        return (manager_handle, storage_handle, size // self._element_size())

    def _shared_decref(self):
        self._untyped_storage._shared_decref()
        return self

    @classmethod
    def _release_ipc_counter(cls, *args, device=None, **kwargs):
        return torch.UntypedStorage._release_ipc_counter_cuda(*args, **kwargs)

    def _shared_incref(self, *args, **kwargs):
        return self._untyped_storage._shared_incref(*args, **kwargs)

    def _share_fd_cpu_(self, *args, **kwargs):
        fd, size = self._untyped_storage._share_fd_cpu_(*args, **kwargs)
        return (fd, size // self._element_size())

    def _get_legacy_storage_class(self):
        if self.dtype not in _dtype_to_storage_type_map():
            return None
        storage_name = _dtype_to_storage_type_map()[self.dtype]
        if self.device.type not in ['cpu', 'cuda', torch._C._get_privateuse1_backend_name()]:
            return None
        module = torch if self.device.type == 'cpu' else getattr(torch, self.device.type)
        try:
            return getattr(module, storage_name)
        except AttributeError:
            return None