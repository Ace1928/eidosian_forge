import sys,operator,logging,traceback
from OpenGL.arrays import _buffers
from OpenGL.raw.GL import _types
from OpenGL.arrays import formathandler
from OpenGL import _configflags
from OpenGL import acceleratesupport
Determine dimensions of the passed array value (if possible)