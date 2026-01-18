import os,sys, logging
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import atexit
class RawOpengl(Widget, Misc):
    """Widget without any sophisticated bindings    by Tom Schwaller"""

    def __init__(self, master=None, cnf={}, **kw):
        Widget.__init__(self, master, 'togl', cnf, kw)
        self.bind('<Map>', self.tkMap)
        self.bind('<Expose>', self.tkExpose)
        self.bind('<Configure>', self.tkExpose)

    def tkRedraw(self, *dummy):
        self.update_idletasks()
        self.tk.call(self._w, 'makecurrent')
        _mode = glGetDoublev(GL_MATRIX_MODE)
        try:
            glMatrixMode(GL_PROJECTION)
            glPushMatrix()
            try:
                self.redraw()
                glFlush()
            finally:
                glPopMatrix()
        finally:
            glMatrixMode(_mode)
        self.tk.call(self._w, 'swapbuffers')

    def tkMap(self, *dummy):
        self.tkExpose()

    def tkExpose(self, *dummy):
        self.tkRedraw()