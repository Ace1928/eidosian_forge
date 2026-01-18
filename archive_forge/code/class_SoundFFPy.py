from kivy.clock import Clock
from kivy.logger import Logger
from kivy.core.audio import Sound, SoundLoader
from kivy.weakmethod import WeakMethod
import time
class SoundFFPy(Sound):

    @staticmethod
    def extensions():
        return formats_in

    def __init__(self, **kwargs):
        self._ffplayer = None
        self.quitted = False
        self._log_callback_set = False
        self._state = ''
        self.state = 'stop'
        if not get_log_callback():
            set_log_callback(_log_callback)
            self._log_callback_set = True
        super(SoundFFPy, self).__init__(**kwargs)

    def __del__(self):
        self.unload()
        if self._log_callback_set:
            set_log_callback(None)

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

    def load(self):
        self.unload()
        ff_opts = {'vn': True, 'sn': True}
        self._ffplayer = MediaPlayer(self.source, callback=self._player_callback, loglevel='info', ff_opts=ff_opts)
        player = self._ffplayer
        player.set_volume(self.volume)
        player.toggle_pause()
        self._state = 'paused'
        s = time.perf_counter()
        while player.get_metadata()['duration'] is None and (not self.quitted) and (time.perf_counter() - s < 10.0):
            time.sleep(0.005)

    def unload(self):
        if self._ffplayer:
            self._ffplayer = None
        self._state = ''
        self.state = 'stop'
        self.quitted = False

    def play(self):
        if self._state == 'playing':
            super(SoundFFPy, self).play()
            return
        if not self._ffplayer:
            self.load()
        self._ffplayer.toggle_pause()
        self._state = 'playing'
        self.state = 'play'
        super(SoundFFPy, self).play()
        self.seek(0)

    def stop(self):
        if self._ffplayer and self._state == 'playing':
            self._ffplayer.toggle_pause()
            self._state = 'paused'
            self.state = 'stop'
        super(SoundFFPy, self).stop()

    def seek(self, position):
        if self._ffplayer is None:
            return
        self._ffplayer.seek(position, relative=False)

    def get_pos(self):
        if self._ffplayer is not None:
            return self._ffplayer.get_pts()
        return 0

    def on_volume(self, instance, volume):
        if self._ffplayer is not None:
            self._ffplayer.set_volume(volume)

    def _get_length(self):
        if self._ffplayer is None:
            return super(SoundFFPy, self)._get_length()
        return self._ffplayer.get_metadata()['duration']

    def _do_eos(self, *args):
        if not self.loop:
            self.stop()
        else:
            self.seek(0.0)