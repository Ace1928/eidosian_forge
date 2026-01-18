import logging
import os
import sys
import threading
from typing import Any, Dict, List, Optional, Tuple
import ray._private.ray_constants as ray_constants
from ray._private.client_mode_hook import (
from ray._private.ray_logging import setup_logger
from ray.job_config import JobConfig
from ray.util.annotations import DeveloperAPI
@DeveloperAPI
class RayAPIStub:
    """This class stands in as the replacement API for the `import ray` module.

    Much like the ray module, this mostly delegates the work to the
    _client_worker. As parts of the ray API are covered, they are piped through
    here or on the client worker API.
    """

    def __init__(self):
        self._cxt = threading.local()
        self._cxt.handler = _default_context
        self._inside_client_test = False

    def get_context(self):
        try:
            return self._cxt.__getattribute__('handler')
        except AttributeError:
            self._cxt.handler = _default_context
            return self._cxt.handler

    def set_context(self, cxt):
        old_cxt = self.get_context()
        if cxt is None:
            self._cxt.handler = _ClientContext()
        else:
            self._cxt.handler = cxt
        return old_cxt

    def is_default(self):
        return self.get_context() == _default_context

    def connect(self, *args, **kw_args):
        self.get_context()._inside_client_test = self._inside_client_test
        conn = self.get_context().connect(*args, **kw_args)
        global _lock, _all_contexts
        with _lock:
            _all_contexts.add(self._cxt.handler)
        return conn

    def disconnect(self, *args, **kw_args):
        global _lock, _all_contexts, _default_context
        with _lock:
            if _default_context == self.get_context():
                for cxt in _all_contexts:
                    cxt.disconnect(*args, **kw_args)
                _all_contexts = set()
            else:
                self.get_context().disconnect(*args, **kw_args)
            if self.get_context() in _all_contexts:
                _all_contexts.remove(self.get_context())
            if len(_all_contexts) == 0:
                _explicitly_disable_client_mode()

    def remote(self, *args, **kwargs):
        return self.get_context().remote(*args, **kwargs)

    def __getattr__(self, name):
        return self.get_context().__getattr__(name)

    def is_connected(self, *args, **kwargs):
        return self.get_context().is_connected(*args, **kwargs)

    def init(self, *args, **kwargs):
        ret = self.get_context().init(*args, **kwargs)
        global _lock, _all_contexts
        with _lock:
            _all_contexts.add(self._cxt.handler)
        return ret

    def shutdown(self, *args, **kwargs):
        global _lock, _all_contexts
        with _lock:
            if _default_context == self.get_context():
                for cxt in _all_contexts:
                    cxt.shutdown(*args, **kwargs)
                _all_contexts = set()
            else:
                self.get_context().shutdown(*args, **kwargs)
            if self.get_context() in _all_contexts:
                _all_contexts.remove(self.get_context())
            if len(_all_contexts) == 0:
                _explicitly_disable_client_mode()