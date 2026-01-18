import collections
import os
import time
from threading import Lock
import glfw
import imageio
import mujoco
import numpy as np
def _import_egl(width, height):
    from mujoco.egl import GLContext
    return GLContext(width, height)