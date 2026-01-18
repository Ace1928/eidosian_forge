from abc import ABCMeta
import copyreg
import functools
import io
import os
import pickle
import socket
import sys
from . import context
def DupFd(fd):
    """Return a wrapper for an fd."""
    popen_obj = context.get_spawning_popen()
    if popen_obj is not None:
        return popen_obj.DupFd(popen_obj.duplicate_for_child(fd))
    elif HAVE_SEND_HANDLE:
        from . import resource_sharer
        return resource_sharer.DupFd(fd)
    else:
        raise ValueError('SCM_RIGHTS appears not to be available')