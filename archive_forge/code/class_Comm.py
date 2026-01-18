import uuid
from typing import Optional
from warnings import warn
import comm.base_comm
import traitlets.config
from traitlets import Bool, Bytes, Instance, Unicode, default
from ipykernel.jsonutil import json_clean
from ipykernel.kernelbase import Kernel
class Comm(BaseComm, traitlets.config.LoggingConfigurable):
    """Class for communicating between a Frontend and a Kernel"""
    kernel = Instance('ipykernel.kernelbase.Kernel', allow_none=True)
    comm_id = Unicode()
    primary = Bool(True, help='Am I the primary or secondary Comm?')
    target_name = Unicode('comm')
    target_module = Unicode(None, allow_none=True, help='requirejs module from\n        which to load comm target.')
    topic = Bytes()

    @default('kernel')
    def _default_kernel(self):
        if Kernel.initialized():
            return Kernel.instance()
        return None

    @default('comm_id')
    def _default_comm_id(self):
        return uuid.uuid4().hex

    def __init__(self, target_name='', data=None, metadata=None, buffers=None, show_warning=True, **kwargs):
        """Initialize a comm."""
        if show_warning:
            warn('The `ipykernel.comm.Comm` class has been deprecated. Please use the `comm` module instead.For creating comms, use the function `from comm import create_comm`.', DeprecationWarning, stacklevel=2)
        had_kernel = 'kernel' in kwargs
        kernel = kwargs.pop('kernel', None)
        if target_name:
            kwargs['target_name'] = target_name
        BaseComm.__init__(self, data=data, metadata=metadata, buffers=buffers, **kwargs)
        if had_kernel:
            kwargs['kernel'] = kernel
        traitlets.config.LoggingConfigurable.__init__(self, **kwargs)