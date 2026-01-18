from OpenGL.platform import CurrentContextIsValid, GLUT_GUARD_CALLBACKS, PLATFORM
from OpenGL import contextdata, error, platform, logs
from OpenGL.raw import GLUT as _simple
from OpenGL._bytes import bytes, unicode,as_8_bit
import ctypes, os, sys, traceback
from OpenGL._bytes import long, integer_types
class GLUTTimerCallback(GLUTCallback):
    """GLUT timer callbacks (completely nonstandard wrt other GLUT callbacks)"""

    def __call__(self, milliseconds, function, value):
        cCallback = self.callbackType(function)
        callbacks = contextdata.getValue(self.CONTEXT_DATA_KEY)
        if callbacks is None:
            callbacks = []
            contextdata.setValue(self.CONTEXT_DATA_KEY, callbacks)

        def deregister(value):
            try:
                function(value)
            finally:
                for item in callbacks:
                    if item.function is deregister:
                        callbacks.remove(item)
                        item.function = None
                        break
                if not callbacks:
                    contextdata.delValue(self.CONTEXT_DATA_KEY)
        cCallback = self.callbackType(deregister)
        cCallback.function = deregister
        callbacks.append(cCallback)
        self.wrappedOperation(milliseconds, cCallback, value)
        return cCallback