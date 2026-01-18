import collections
import os
import time
from threading import Lock
import glfw
import imageio
import mujoco
import numpy as np
def _import_osmesa(width, height):
    from mujoco.osmesa import GLContext
    return GLContext(width, height)