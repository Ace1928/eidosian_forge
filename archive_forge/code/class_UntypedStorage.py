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
class UntypedStorage(torch._C.StorageBase, _StorageBase):

    def __getitem__(self, *args, **kwargs):
        if self.device.type == 'meta':
            raise NotImplementedError("Not available for 'meta' device type")
        return super().__getitem__(*args, **kwargs)

    @property
    def is_cuda(self):
        return self.device.type == 'cuda'

    @property
    def is_hpu(self):
        return self.device.type == 'hpu'

    @property
    def filename(self) -> _Optional[str]:
        """Returns the file name associated with this storage if the storage was memory mapped from a file.
           or ``None`` if the storage was not created by memory mapping a file."""
        return self._get_filename()

    @_share_memory_lock_protected
    def share_memory_(self, *args, **kwargs):
        """
        Moves the storage to shared memory.

        This is a no-op for storages already in shared memory and for CUDA
        storages, which do not need to be moved for sharing across processes.
        Storages in shared memory cannot be resized.

        Note that to mitigate issues like `this <https://github.com/pytorch/pytorch/issues/95606>`_
        it is thread safe to call this function from multiple threads on the same object.
        It is NOT thread safe though to call any other function on self without proper
        synchronization. Please see :doc:`/notes/multiprocessing` for more details.

        .. note::
            When all references to a storage in shared memory are deleted, the associated shared memory
            object will also be deleted. PyTorch has a special cleanup process to ensure that this happens
            even if the current process exits unexpectedly.

            It is worth noting the difference between :meth:`share_memory_` and :meth:`from_file` with ``shared = True``

            #. ``share_memory_`` uses `shm_open(3) <https://man7.org/linux/man-pages/man3/shm_open.3.html>`_ to create a
               POSIX shared memory object while :meth:`from_file` uses
               `open(2) <https://man7.org/linux/man-pages/man2/open.2.html>`_ to open the filename passed by the user.
            #. Both use an `mmap(2) call <https://man7.org/linux/man-pages/man2/mmap.2.html>`_ with ``MAP_SHARED``
               to map the file/object into the current virtual address space
            #. ``share_memory_`` will call ``shm_unlink(3)`` on the object after mapping it to make sure the shared memory
               object is freed when no process has the object open. ``torch.from_file(shared=True)`` does not unlink the
               file. This file is persistent and will remain until it is deleted by the user.

        Returns:
            ``self``
        """
        return super().share_memory_(*args, **kwargs)

    @_share_memory_lock_protected
    def _share_fd_cpu_(self, *args, **kwargs):
        return super()._share_fd_cpu_(*args, **kwargs)

    @_share_memory_lock_protected
    def _share_filename_cpu_(self, *args, **kwargs):
        return super()._share_filename_cpu_(*args, **kwargs)