from kivy.lib.gstplayer import GstPlayer, get_gst_version
from kivy.core.audio import Sound, SoundLoader
from kivy.logger import Logger
from kivy.compat import PY2
from kivy.clock import Clock
from os.path import realpath
class SoundGstplayer(Sound):

    @staticmethod
    def extensions():
        return ('wav', 'ogg', 'mp3', 'm4a', 'flac', 'mp4')

    def __init__(self, **kwargs):
        self.player = None
        super(SoundGstplayer, self).__init__(**kwargs)

    def _on_gst_eos_sync(self):
        Clock.schedule_once(self._on_gst_eos, 0)

    def _on_gst_eos(self, *dt):
        if self.loop:
            self.player.stop()
            self.player.play()
        else:
            self.stop()

    def load(self):
        self.unload()
        uri = self._get_uri()
        self.player = GstPlayer(uri, None, self._on_gst_eos_sync, _on_gstplayer_message)
        self.player.load()

    def play(self):
        self.player.set_volume(self.volume)
        self.player.play()
        super(SoundGstplayer, self).play()

    def stop(self):
        self.player.stop()
        super(SoundGstplayer, self).stop()

    def unload(self):
        if self.player:
            self.player.unload()
            self.player = None

    def seek(self, position):
        self.player.seek(position / self.length)

    def get_pos(self):
        return self.player.get_position()

    def _get_length(self):
        return self.player.get_duration()

    def on_volume(self, instance, volume):
        self.player.set_volume(volume)

    def _get_uri(self):
        uri = self.source
        if not uri:
            return
        if '://' not in uri:
            uri = 'file:' + pathname2url(realpath(uri))
        return uri