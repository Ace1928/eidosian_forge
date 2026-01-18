from __future__ import annotations
import io
import logging
import os
import threading
import warnings
import weakref
from errno import ESPIPE
from glob import has_magic
from hashlib import sha256
from typing import ClassVar
from .callbacks import DEFAULT_CALLBACK
from .config import apply_config, conf
from .dircache import DirCache
from .transaction import Transaction
from .utils import (
class _Cached(type):
    """
    Metaclass for caching file system instances.

    Notes
    -----
    Instances are cached according to

    * The values of the class attributes listed in `_extra_tokenize_attributes`
    * The arguments passed to ``__init__``.

    This creates an additional reference to the filesystem, which prevents the
    filesystem from being garbage collected when all *user* references go away.
    A call to the :meth:`AbstractFileSystem.clear_instance_cache` must *also*
    be made for a filesystem instance to be garbage collected.
    """

    def __init__(cls, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if conf.get('weakref_instance_cache'):
            cls._cache = weakref.WeakValueDictionary()
        else:
            cls._cache = {}
        cls._pid = os.getpid()

    def __call__(cls, *args, **kwargs):
        kwargs = apply_config(cls, kwargs)
        extra_tokens = tuple((getattr(cls, attr, None) for attr in cls._extra_tokenize_attributes))
        token = tokenize(cls, cls._pid, threading.get_ident(), *args, *extra_tokens, **kwargs)
        skip = kwargs.pop('skip_instance_cache', False)
        if os.getpid() != cls._pid:
            cls._cache.clear()
            cls._pid = os.getpid()
        if not skip and cls.cachable and (token in cls._cache):
            cls._latest = token
            return cls._cache[token]
        else:
            obj = super().__call__(*args, **kwargs)
            obj._fs_token_ = token
            obj.storage_args = args
            obj.storage_options = kwargs
            if obj.async_impl and obj.mirror_sync_methods:
                from .asyn import mirror_sync_methods
                mirror_sync_methods(obj)
            if cls.cachable and (not skip):
                cls._latest = token
                cls._cache[token] = obj
            return obj