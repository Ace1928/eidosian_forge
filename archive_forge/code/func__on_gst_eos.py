from kivy.lib.gstplayer import GstPlayer, get_gst_version
from kivy.core.audio import Sound, SoundLoader
from kivy.logger import Logger
from kivy.compat import PY2
from kivy.clock import Clock
from os.path import realpath
def _on_gst_eos(self, *dt):
    if self.loop:
        self.player.stop()
        self.player.play()
    else:
        self.stop()