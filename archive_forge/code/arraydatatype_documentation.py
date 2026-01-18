import ctypes
import OpenGL
from OpenGL.raw.GL import _types
from OpenGL import plugins
from OpenGL.arrays import formathandler, _arrayconstants as GL_1_1
from OpenGL import logs
from OpenGL import acceleratesupport
Given a data-value, try to determine number of bytes it's final form occupies

            For most data-types this is arraySize() * atomic-unit-size
            