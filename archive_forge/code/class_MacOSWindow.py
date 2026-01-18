import Quartz
import pygetwindow
class MacOSWindow:

    def __init__(self, hWnd):
        self._hWnd = hWnd

        def _onRead(attrName):
            r = self._getWindowRect(_hWnd)
            self._rect._left = r.left
            self._rect._top = r.top
            self._rect._width = r.right - r.left
            self._rect._height = r.bottom - r.top

        def _onChange(oldBox, newBox):
            self.moveTo(newBox.left, newBox.top)
            self.resizeTo(newBox.width, newBox.height)
        r = self._getWindowRect(_hWnd)
        self._rect = pyrect.Rect(r.left, r.top, r.right - r.left, r.bottom - r.top, onChange=_onChange, onRead=_onRead)

    def __str__(self):
        r = self._getWindowRect(_hWnd)
        width = r.right - r.left
        height = r.bottom - r.top
        return '<%s left="%s", top="%s", width="%s", height="%s", title="%s">' % (self.__class__.__name__, r.left, r.top, width, height, self.title)

    def __repr__(self):
        return '%s(hWnd=%s)' % (self.__class__.__name__, self._hWnd)

    def __eq__(self, other):
        return isinstance(other, MacOSWindow) and self._hWnd == other._hWnd

    def close(self):
        """Closes this window. This may trigger "Are you sure you want to
        quit?" dialogs or other actions that prevent the window from
        actually closing. This is identical to clicking the X button on the
        window."""
        raise NotImplementedError

    def minimize(self):
        """Minimizes this window."""
        raise NotImplementedError

    def maximize(self):
        """Maximizes this window."""
        raise NotImplementedError

    def restore(self):
        """If maximized or minimized, restores the window to it's normal size."""
        raise NotImplementedError

    def activate(self):
        """Activate this window and make it the foreground window."""
        raise NotImplementedError

    def resizeRel(self, widthOffset, heightOffset):
        """Resizes the window relative to its current size."""
        raise NotImplementedError

    def resizeTo(self, newWidth, newHeight):
        """Resizes the window to a new width and height."""
        raise NotImplementedError

    def moveRel(self, xOffset, yOffset):
        """Moves the window relative to its current position."""
        raise NotImplementedError

    def moveTo(self, newLeft, newTop):
        """Moves the window to new coordinates on the screen."""
        raise NotImplementedError

    @property
    def isMinimized(self):
        """Returns True if the window is currently minimized."""
        raise NotImplementedError

    @property
    def isMaximized(self):
        """Returns True if the window is currently maximized."""
        raise NotImplementedError

    @property
    def isActive(self):
        """Returns True if the window is currently the active, foreground window."""
        raise NotImplementedError

    @property
    def title(self):
        """Returns the window title as a string."""
        raise NotImplementedError

    @property
    def visible(self):
        raise NotImplementedError