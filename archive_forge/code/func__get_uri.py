from kivy.lib.gstplayer import GstPlayer, get_gst_version
from kivy.core.audio import Sound, SoundLoader
from kivy.logger import Logger
from kivy.compat import PY2
from kivy.clock import Clock
from os.path import realpath
def _get_uri(self):
    uri = self.source
    if not uri:
        return
    if '://' not in uri:
        uri = 'file:' + pathname2url(realpath(uri))
    return uri