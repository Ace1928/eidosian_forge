import collections
import os
import time
from threading import Lock
import glfw
import imageio
import mujoco
import numpy as np
def _get_opengl_backend(self, width, height):
    backend = os.environ.get('MUJOCO_GL')
    if backend is not None:
        try:
            self.opengl_context = _ALL_RENDERERS[backend](width, height)
        except KeyError:
            raise RuntimeError('Environment variable {} must be one of {!r}: got {!r}.'.format('MUJOCO_GL', _ALL_RENDERERS.keys(), backend))
    else:
        for name, _ in _ALL_RENDERERS.items():
            try:
                self.opengl_context = _ALL_RENDERERS[name](width, height)
                backend = name
                break
            except:
                pass
        if backend is None:
            raise RuntimeError('No OpenGL backend could be imported. Attempting to create a rendering context will result in a RuntimeError.')