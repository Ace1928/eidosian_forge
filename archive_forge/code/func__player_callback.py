from kivy.clock import Clock
from kivy.logger import Logger
from kivy.core.audio import Sound, SoundLoader
from kivy.weakmethod import WeakMethod
import time
def _player_callback(self, selector, value):
    if self._ffplayer is None:
        return
    if selector == 'quit':

        def close(*args):
            self.quitted = True
            self.unload()
        Clock.schedule_once(close, 0)
    elif selector == 'eof':
        Clock.schedule_once(self._do_eos, 0)