import weakref
from enum import Enum
import threading
from typing import Tuple
import pyglet
from pyglet import gl
from pyglet.gl import gl_info
def _delete_objects_one_by_one(self, list_, deletion_func):
    """Similar to ``_delete_objects``, but assumes the deletion functions's
        signature to be ``(GLuint name)``, calling it once for each object.
        """
    count = len(list_)
    to_delete = list_[:count]
    del list_[:count]
    for name in to_delete:
        deletion_func(gl.GLuint(name))