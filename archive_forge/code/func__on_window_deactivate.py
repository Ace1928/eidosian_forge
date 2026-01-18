from typing import Optional
import time
from collections import defaultdict
import gym
from gym import spaces
import pyglet
import pyglet.window.key as key
def _on_window_deactivate(self):
    self.window.set_mouse_visible(True)
    self.window.set_exclusive_mouse(False)