from kivy.lib.gstplayer import GstPlayer, get_gst_version
from kivy.core.audio import Sound, SoundLoader
from kivy.logger import Logger
from kivy.compat import PY2
from kivy.clock import Clock
from os.path import realpath

Audio Gstplayer
===============

.. versionadded:: 1.8.0

Implementation of a VideoBase with Kivy :class:`~kivy.lib.gstplayer.GstPlayer`
This player is the preferred player, using Gstreamer 1.0, working on both
Python 2 and 3.
