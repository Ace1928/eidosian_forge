from typing import Optional
import time
from collections import defaultdict
import gym
from gym import spaces
import pyglet
import pyglet.window.key as key
def _on_key_press(self, symbol, modifiers):
    self.pressed_keys[symbol] = True