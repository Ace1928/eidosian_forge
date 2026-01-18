import array
from twisted.conch.insults import helper, insults
from twisted.python import text as tptext
class TextOutput(Widget):
    text = b''

    def __init__(self, size=None):
        Widget.__init__(self)
        self.size = size

    def sizeHint(self):
        return self.size

    def render(self, width, height, terminal):
        terminal.cursorPosition(0, 0)
        text = self.text[:width]
        terminal.write(text + b' ' * (width - len(text)))

    def setText(self, text):
        self.text = text
        self.repaint()

    def focusReceived(self):
        raise YieldFocus()