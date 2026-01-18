import collections
import os
import time
from threading import Lock
import glfw
import imageio
import mujoco
import numpy as np
def _mouse_button_callback(self, window, button, act, mods):
    self._button_left_pressed = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
    self._button_right_pressed = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
    x, y = glfw.get_cursor_pos(window)
    self._last_mouse_x = int(self._scale * x)
    self._last_mouse_y = int(self._scale * y)