from kivy.logger import Logger
from kivy.event import EventDispatcher
from kivy.core import core_register_libs
from kivy.resources import resource_find
from kivy.properties import StringProperty, NumericProperty, OptionProperty, \
from kivy.utils import platform
from kivy.setupconfig import USE_SDL2
from sys import float_info
class SoundLoader:
    """Load a sound, using the best loader for the given file type.
    """
    _classes = []

    @staticmethod
    def register(classobj):
        """Register a new class to load the sound."""
        Logger.debug('Audio: register %s' % classobj.__name__)
        SoundLoader._classes.append(classobj)

    @staticmethod
    def load(filename):
        """Load a sound, and return a Sound() instance."""
        rfn = resource_find(filename)
        if rfn is not None:
            filename = rfn
        ext = filename.split('.')[-1].lower()
        if '?' in ext:
            ext = ext.split('?')[0]
        for classobj in SoundLoader._classes:
            if ext in classobj.extensions():
                return classobj(source=filename)
        Logger.warning('Audio: Unable to find a loader for <%s>' % filename)
        return None