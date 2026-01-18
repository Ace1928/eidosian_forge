import functools
import threading
from contextlib import contextmanager
from .driver import driver, USE_NV_BINDING
class _Runtime(object):
    """Emulate the CUDA runtime context management.

    It owns all Devices and Contexts.
    Keeps at most one Context per Device
    """

    def __init__(self):
        self.gpus = _DeviceList()
        self._tls = threading.local()
        self._mainthread = threading.current_thread()
        self._lock = threading.RLock()

    @contextmanager
    def ensure_context(self):
        """Ensure a CUDA context is available inside the context.

        On entrance, queries the CUDA driver for an active CUDA context and
        attaches it in TLS for subsequent calls so they do not need to query
        the CUDA driver again.  On exit, detach the CUDA context from the TLS.

        This will allow us to pickup thirdparty activated CUDA context in
        any top-level Numba CUDA API.
        """
        with driver.get_active_context():
            oldctx = self._get_attached_context()
            newctx = self.get_or_create_context(None)
            self._set_attached_context(newctx)
            try:
                yield
            finally:
                self._set_attached_context(oldctx)

    def get_or_create_context(self, devnum):
        """Returns the primary context and push+create it if needed
        for *devnum*.  If *devnum* is None, use the active CUDA context (must
        be primary) or create a new one with ``devnum=0``.
        """
        if devnum is None:
            attached_ctx = self._get_attached_context()
            if attached_ctx is None:
                return self._get_or_create_context_uncached(devnum)
            else:
                return attached_ctx
        else:
            if USE_NV_BINDING:
                devnum = int(devnum)
            return self._activate_context_for(devnum)

    def _get_or_create_context_uncached(self, devnum):
        """See also ``get_or_create_context(devnum)``.
        This version does not read the cache.
        """
        with self._lock:
            with driver.get_active_context() as ac:
                if not ac:
                    return self._activate_context_for(0)
                else:
                    ctx = self.gpus[ac.devnum].get_primary_context()
                    if USE_NV_BINDING:
                        ctx_handle = int(ctx.handle)
                        ac_ctx_handle = int(ac.context_handle)
                    else:
                        ctx_handle = ctx.handle.value
                        ac_ctx_handle = ac.context_handle.value
                    if ctx_handle != ac_ctx_handle:
                        msg = 'Numba cannot operate on non-primary CUDA context {:x}'
                        raise RuntimeError(msg.format(ac_ctx_handle))
                    ctx.prepare_for_use()
                return ctx

    def _activate_context_for(self, devnum):
        with self._lock:
            gpu = self.gpus[devnum]
            newctx = gpu.get_primary_context()
            cached_ctx = self._get_attached_context()
            if cached_ctx is not None and cached_ctx is not newctx:
                raise RuntimeError('Cannot switch CUDA-context.')
            newctx.push()
            return newctx

    def _get_attached_context(self):
        return getattr(self._tls, 'attached_context', None)

    def _set_attached_context(self, ctx):
        self._tls.attached_context = ctx

    def reset(self):
        """Clear all contexts in the thread.  Destroy the context if and only
        if we are in the main thread.
        """
        while driver.pop_active_context() is not None:
            pass
        if threading.current_thread() == self._mainthread:
            self._destroy_all_contexts()

    def _destroy_all_contexts(self):
        for gpu in self.gpus:
            gpu.reset()