from kivy.logger import Logger
from kivy.event import EventDispatcher
from kivy.core import core_register_libs
from kivy.resources import resource_find
from kivy.properties import StringProperty, NumericProperty, OptionProperty, \
from kivy.utils import platform
from kivy.setupconfig import USE_SDL2
from sys import float_info
class Sound(EventDispatcher):
    """Represents a sound to play. This class is abstract, and cannot be used
    directly.

    Use SoundLoader to load a sound.

    :Events:
        `on_play`: None
            Fired when the sound is played.
        `on_stop`: None
            Fired when the sound is stopped.
    """
    source = StringProperty(None)
    'Filename / source of your audio file.\n\n    .. versionadded:: 1.3.0\n\n    :attr:`source` is a :class:`~kivy.properties.StringProperty` that defaults\n    to None and is read-only. Use the :meth:`SoundLoader.load` for loading\n    audio.\n    '
    volume = NumericProperty(1.0)
    'Volume, in the range 0-1. 1 means full volume, 0 means mute.\n\n    .. versionadded:: 1.3.0\n\n    :attr:`volume` is a :class:`~kivy.properties.NumericProperty` and defaults\n    to 1.\n    '
    pitch = BoundedNumericProperty(1.0, min=float_info.epsilon)
    'Pitch of a sound. 2 is an octave higher, .5 one below. This is only\n    implemented for SDL2 audio provider yet.\n\n    .. versionadded:: 1.10.0\n\n    :attr:`pitch` is a :class:`~kivy.properties.NumericProperty` and defaults\n    to 1.\n    '
    state = OptionProperty('stop', options=('stop', 'play'))
    "State of the sound, one of 'stop' or 'play'.\n\n    .. versionadded:: 1.3.0\n\n    :attr:`state` is a read-only :class:`~kivy.properties.OptionProperty`."
    loop = BooleanProperty(False)
    'Set to True if the sound should automatically loop when it finishes.\n\n    .. versionadded:: 1.8.0\n\n    :attr:`loop` is a :class:`~kivy.properties.BooleanProperty` and defaults to\n    False.'
    __events__ = ('on_play', 'on_stop')

    def on_source(self, instance, filename):
        self.unload()
        if filename is None:
            return
        self.load()

    def get_pos(self):
        """
        Returns the current position of the audio file.
        Returns 0 if not playing.

        .. versionadded:: 1.4.1
        """
        return 0

    def _get_length(self):
        return 0
    length = property(lambda self: self._get_length(), doc='Get length of the sound (in seconds).')

    def load(self):
        """Load the file into memory."""
        pass

    def unload(self):
        """Unload the file from memory."""
        pass

    def play(self):
        """Play the file."""
        self.state = 'play'
        self.dispatch('on_play')

    def stop(self):
        """Stop playback."""
        self.state = 'stop'
        self.dispatch('on_stop')

    def seek(self, position):
        """Go to the <position> (in seconds).

        .. note::
            Most sound providers cannot seek when the audio is stopped.
            Play then seek.
        """
        pass

    def on_play(self):
        pass

    def on_stop(self):
        pass