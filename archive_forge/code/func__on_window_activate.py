from typing import Optional
import time
from collections import defaultdict
import gym
from gym import spaces
import pyglet
import pyglet.window.key as key
def _on_window_activate(self):
    self.window.set_mouse_visible(False)
    self.window.set_exclusive_mouse(True)