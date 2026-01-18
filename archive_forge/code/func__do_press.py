from kivy.clock import Clock
from kivy.config import Config
from kivy.properties import OptionProperty, ObjectProperty, \
from time import time
def _do_press(self):
    self.state = 'down'