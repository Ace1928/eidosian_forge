import os
import sys
import threading
import time
import traceback
import warnings
import weakref
import builtins
import pickle
import numpy as np
from ..util import cprint
class LocalObjectProxy(object):
    """
    Used for wrapping local objects to ensure that they are send by proxy to a remote host.
    Note that 'proxy' is just a shorter alias for LocalObjectProxy.
    
    For example::
    
        data = [1,2,3,4,5]
        remotePlot.plot(data)         ## by default, lists are pickled and sent by value
        remotePlot.plot(proxy(data))  ## force the object to be sent by proxy
    
    """
    nextProxyId = 0
    proxiedObjects = {}

    @classmethod
    def registerObject(cls, obj):
        pid = cls.nextProxyId
        cls.nextProxyId += 1
        cls.proxiedObjects[pid] = obj
        return pid

    @classmethod
    def lookupProxyId(cls, pid):
        return cls.proxiedObjects[pid]

    @classmethod
    def releaseProxyId(cls, pid):
        del cls.proxiedObjects[pid]

    def __init__(self, obj, **opts):
        """
        Create a 'local' proxy object that, when sent to a remote host,
        will appear as a normal ObjectProxy to *obj*. 
        Any extra keyword arguments are passed to proxy._setProxyOptions()
        on the remote side.
        """
        self.processId = os.getpid()
        self.typeStr = repr(obj)
        self.obj = obj
        self.opts = opts

    def __reduce__(self):
        pid = LocalObjectProxy.registerObject(self.obj)
        return (unpickleObjectProxy, (self.processId, pid, self.typeStr, None, self.opts))