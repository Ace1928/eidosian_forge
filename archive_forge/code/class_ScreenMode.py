from pyglet import gl
from pyglet import app
from pyglet import window
from pyglet import canvas
class ScreenMode:
    """Screen resolution and display settings.

    Applications should not construct `ScreenMode` instances themselves; see
    :meth:`Screen.get_modes`.

    The :attr:`depth` and :attr:`rate` variables may be ``None`` if the 
    operating system does not provide relevant data.

    .. versionadded:: 1.2

    """
    width = None
    'Width of screen, in pixels.\n\n    :type: int\n    '
    height = None
    'Height of screen, in pixels.\n\n    :type: int\n    '
    depth = None
    'Pixel color depth, in bits per pixel.\n\n    :type: int\n    '
    rate = None
    'Screen refresh rate in Hz.\n\n    :type: int\n    '

    def __init__(self, screen):
        """
        
        :parameters:
            `screen` : `Screen`
        """
        self.screen = screen

    def __repr__(self):
        return f'{self.__class__.__name__}(width={self.width!r}, height={self.height!r}, depth={self.depth!r}, rate={self.rate})'