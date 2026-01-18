import array
from twisted.conch.insults import helper, insults
from twisted.python import text as tptext
class ScrolledArea(Widget):
    """
    A L{ScrolledArea} contains another widget wrapped in a viewport and
    vertical and horizontal scrollbars for moving the viewport around.
    """

    def __init__(self, containee):
        Widget.__init__(self)
        self._viewport = Viewport(containee)
        self._horiz = HorizontalScrollbar(self._horizScroll)
        self._vert = VerticalScrollbar(self._vertScroll)
        for w in (self._viewport, self._horiz, self._vert):
            w.parent = self

    def _horizScroll(self, n):
        self._viewport.xOffset += n
        self._viewport.xOffset = max(0, self._viewport.xOffset)
        return self._viewport.xOffset / 25.0

    def _vertScroll(self, n):
        self._viewport.yOffset += n
        self._viewport.yOffset = max(0, self._viewport.yOffset)
        return self._viewport.yOffset / 25.0

    def func_UP_ARROW(self, modifier):
        self._vert.smaller()

    def func_DOWN_ARROW(self, modifier):
        self._vert.bigger()

    def func_LEFT_ARROW(self, modifier):
        self._horiz.smaller()

    def func_RIGHT_ARROW(self, modifier):
        self._horiz.bigger()

    def filthy(self):
        self._viewport.filthy()
        self._horiz.filthy()
        self._vert.filthy()
        Widget.filthy(self)

    def render(self, width, height, terminal):
        wrapper = BoundedTerminalWrapper(terminal, width - 2, height - 2, 1, 1)
        self._viewport.draw(width - 2, height - 2, wrapper)
        if self.focused:
            terminal.write(b'\x1b[31m')
        horizontalLine(terminal, 0, 1, width - 1)
        verticalLine(terminal, 0, 1, height - 1)
        self._vert.draw(1, height - 1, BoundedTerminalWrapper(terminal, 1, height - 1, width - 1, 0))
        self._horiz.draw(width, 1, BoundedTerminalWrapper(terminal, width, 1, 0, height - 1))
        terminal.write(b'\x1b[0m')