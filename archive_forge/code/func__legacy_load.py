import difflib
import os
import io
import shutil
import struct
import sys
import torch
import tarfile
import tempfile
import warnings
from contextlib import closing, contextmanager
from enum import Enum
from ._utils import _import_dotted_name
from torch._sources import get_source_lines_and_file
from torch.types import Storage
from torch.storage import _get_dtype_from_pickle_storage_type
from typing import Any, BinaryIO, Callable, cast, Dict, Optional, Type, Tuple, Union, IO
from typing_extensions import TypeAlias  # Python 3.10+
import copyreg
import pickle
import pathlib
import torch._weights_only_unpickler as _weights_only_unpickler
def _legacy_load(f, map_location, pickle_module, **pickle_load_args):
    deserialized_objects: Dict[int, Any] = {}
    restore_location = _get_restore_location(map_location)

    class UnpicklerWrapper(pickle_module.Unpickler):

        def find_class(self, mod_name, name):
            if type(name) is str and 'Storage' in name:
                try:
                    return StorageType(name)
                except KeyError:
                    pass
            return super().find_class(mod_name, name)

    def _check_container_source(container_type, source_file, original_source):
        try:
            current_source = ''.join(get_source_lines_and_file(container_type)[0])
        except Exception:
            warnings.warn("Couldn't retrieve source code for container of type " + container_type.__name__ + ". It won't be checked for correctness upon loading.")
            return
        if original_source != current_source:
            if container_type.dump_patches:
                file_name = container_type.__name__ + '.patch'
                diff = difflib.unified_diff(current_source.split('\n'), original_source.split('\n'), source_file, source_file, lineterm='')
                lines = '\n'.join(diff)
                try:
                    with open(file_name, 'a+') as f:
                        file_size = f.seek(0, 2)
                        f.seek(0)
                        if file_size == 0:
                            f.write(lines)
                        elif file_size != len(lines) or f.read() != lines:
                            raise OSError
                    msg = 'Saved a reverse patch to ' + file_name + '. Run `patch -p0 < ' + file_name + '` to revert your changes.'
                except OSError:
                    msg = "Tried to save a patch, but couldn't create a writable file " + file_name + ". Make sure it doesn't exist and your working directory is writable."
            else:
                msg = "you can retrieve the original source code by accessing the object's source attribute or set `torch.nn.Module.dump_patches = True` and use the patch tool to revert the changes."
            msg = f"source code of class '{torch.typename(container_type)}' has changed. {msg}"
            warnings.warn(msg, SourceChangeWarning)

    def legacy_load(f):
        deserialized_objects: Dict[int, Any] = {}

        def persistent_load(saved_id):
            if isinstance(saved_id, tuple):
                if all(saved_id[1:]):
                    _check_container_source(*saved_id)
                return saved_id[0]
            return deserialized_objects[int(saved_id)]
        with closing(tarfile.open(fileobj=f, mode='r:', format=tarfile.PAX_FORMAT)) as tar, mkdtemp() as tmpdir:
            tar.extract('storages', path=tmpdir)
            with open(os.path.join(tmpdir, 'storages'), 'rb', 0) as f:
                num_storages = pickle_module.load(f, **pickle_load_args)
                for i in range(num_storages):
                    args = pickle_module.load(f, **pickle_load_args)
                    key, location, storage_type = args
                    dtype = storage_type._dtype
                    obj = cast(Storage, torch.UntypedStorage)._new_with_file(f, torch._utils._element_size(dtype))
                    obj = restore_location(obj, location)
                    deserialized_objects[key] = torch.storage.TypedStorage(wrap_storage=obj, dtype=dtype, _internal=True)
                storage_views = pickle_module.load(f, **pickle_load_args)
                for target_cdata, root_cdata, offset, numel in storage_views:
                    root = deserialized_objects[root_cdata]
                    element_size = torch._utils._element_size(root.dtype)
                    offset_bytes = offset * element_size
                    deserialized_objects[target_cdata] = torch.storage.TypedStorage(wrap_storage=root._untyped_storage[offset_bytes:offset_bytes + numel * element_size], dtype=root.dtype, _internal=True)
            tar.extract('tensors', path=tmpdir)
            with open(os.path.join(tmpdir, 'tensors'), 'rb', 0) as f:
                num_tensors = pickle_module.load(f, **pickle_load_args)
                for _ in range(num_tensors):
                    args = pickle_module.load(f, **pickle_load_args)
                    key, storage_id, original_tensor_type = args
                    storage = deserialized_objects[storage_id]
                    ndim, = struct.unpack('<i', f.read(4))
                    f.read(4)
                    numel = struct.unpack(f'<{ndim}q', f.read(8 * ndim))
                    stride = struct.unpack(f'<{ndim}q', f.read(8 * ndim))
                    storage_offset, = struct.unpack('<q', f.read(8))
                    tensor = torch.tensor([], dtype=storage.dtype).set_(storage._untyped_storage, storage_offset, numel, stride)
                    deserialized_objects[key] = tensor
            pickle_file = tar.extractfile('pickle')
            unpickler = UnpicklerWrapper(pickle_file, **pickle_load_args)
            unpickler.persistent_load = persistent_load
            result = unpickler.load()
            return result
    deserialized_objects = {}

    def persistent_load(saved_id):
        assert isinstance(saved_id, tuple)
        typename = _maybe_decode_ascii(saved_id[0])
        data = saved_id[1:]
        if typename == 'module':
            if all(data[1:]):
                _check_container_source(*data)
            return data[0]
        elif typename == 'storage':
            storage_type, root_key, location, numel, view_metadata = data
            location = _maybe_decode_ascii(location)
            dtype = storage_type.dtype
            nbytes = numel * torch._utils._element_size(dtype)
            if root_key not in deserialized_objects:
                obj = cast(Storage, torch.UntypedStorage(nbytes))
                obj._torch_load_uninitialized = True
                typed_storage = torch.storage.TypedStorage(wrap_storage=restore_location(obj, location), dtype=dtype, _internal=True)
                deserialized_objects[root_key] = typed_storage
            else:
                typed_storage = deserialized_objects[root_key]
                if typed_storage._data_ptr() == 0:
                    typed_storage = torch.storage.TypedStorage(device=typed_storage._untyped_storage.device, dtype=dtype, _internal=True)
            if view_metadata is not None:
                view_key, offset, view_size = view_metadata
                offset_bytes = offset * torch._utils._element_size(dtype)
                view_size_bytes = view_size * torch._utils._element_size(dtype)
                if view_key not in deserialized_objects:
                    deserialized_objects[view_key] = torch.storage.TypedStorage(wrap_storage=typed_storage._untyped_storage[offset_bytes:offset_bytes + view_size_bytes], dtype=dtype, _internal=True)
                res = deserialized_objects[view_key]
            else:
                res = typed_storage
            return res
        else:
            raise RuntimeError(f'Unknown saved id type: {saved_id[0]}')
    _check_seekable(f)
    f_should_read_directly = _should_read_directly(f)
    if f_should_read_directly and f.tell() == 0:
        try:
            return legacy_load(f)
        except tarfile.TarError:
            if _is_zipfile(f):
                raise RuntimeError(f'{f.name} is a zip archive (did you mean to use torch.jit.load()?)') from None
            f.seek(0)
    if not hasattr(f, 'readinto') and (3, 8, 0) <= sys.version_info < (3, 8, 2):
        raise RuntimeError(f'torch.load does not work with file-like objects that do not implement readinto on Python 3.8.0 and 3.8.1. Received object of type "{type(f)}". Please update to Python 3.8.2 or newer to restore this functionality.')
    magic_number = pickle_module.load(f, **pickle_load_args)
    if magic_number != MAGIC_NUMBER:
        raise RuntimeError('Invalid magic number; corrupt file?')
    protocol_version = pickle_module.load(f, **pickle_load_args)
    if protocol_version != PROTOCOL_VERSION:
        raise RuntimeError(f'Invalid protocol version: {protocol_version}')
    _sys_info = pickle_module.load(f, **pickle_load_args)
    unpickler = UnpicklerWrapper(f, **pickle_load_args)
    unpickler.persistent_load = persistent_load
    result = unpickler.load()
    deserialized_storage_keys = pickle_module.load(f, **pickle_load_args)
    offset = f.tell() if f_should_read_directly else None
    for key in deserialized_storage_keys:
        assert key in deserialized_objects
        typed_storage = deserialized_objects[key]
        typed_storage._untyped_storage._set_from_file(f, offset, f_should_read_directly, torch._utils._element_size(typed_storage.dtype))
        if offset is not None:
            offset = f.tell()
    torch._utils._validate_loaded_sparse_tensors()
    return result