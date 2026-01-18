import sys
import enum
import warnings
import operator
from pyglet.event import EventDispatcher
class Tablet:
    """High-level interface to tablet devices.

    Unlike other devices, tablets must be opened for a specific window,
    and cannot be opened exclusively.  The `open` method returns a
    `TabletCanvas` object, which supports the events provided by the tablet.

    Currently only one tablet device can be used, though it can be opened on
    multiple windows.  If more than one tablet is connected, the behaviour is
    undefined.
    """

    def open(self, window):
        """Open a tablet device for a window.

        :Parameters:
            `window` : `Window`
                The window on which the tablet will be used.

        :rtype: `TabletCanvas`
        """
        raise NotImplementedError('abstract')