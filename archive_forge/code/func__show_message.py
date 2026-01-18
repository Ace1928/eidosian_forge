from typing import Optional
import time
from collections import defaultdict
import gym
from gym import spaces
import pyglet
import pyglet.window.key as key
def _show_message(self, text):
    label = pyglet.text.Label(text, font_size=32, x=self.window.width // 2, y=self.window.height // 2, anchor_x='center', anchor_y='center')
    label.draw()
    self.window.flip()