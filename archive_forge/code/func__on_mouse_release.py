from typing import Optional
import time
from collections import defaultdict
import gym
from gym import spaces
import pyglet
import pyglet.window.key as key
def _on_mouse_release(self, x, y, button, modifiers):
    self.pressed_keys[button] = False