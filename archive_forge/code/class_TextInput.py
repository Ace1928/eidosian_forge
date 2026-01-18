import array
from twisted.conch.insults import helper, insults
from twisted.python import text as tptext
class TextInput(Widget):

    def __init__(self, maxwidth, onSubmit):
        Widget.__init__(self)
        self.onSubmit = onSubmit
        self.maxwidth = maxwidth
        self.buffer = b''
        self.cursor = 0

    def setText(self, text):
        self.buffer = text[:self.maxwidth]
        self.cursor = len(self.buffer)
        self.repaint()

    def func_LEFT_ARROW(self, modifier):
        if self.cursor > 0:
            self.cursor -= 1
            self.repaint()

    def func_RIGHT_ARROW(self, modifier):
        if self.cursor < len(self.buffer):
            self.cursor += 1
            self.repaint()

    def backspaceReceived(self):
        if self.cursor > 0:
            self.buffer = self.buffer[:self.cursor - 1] + self.buffer[self.cursor:]
            self.cursor -= 1
            self.repaint()

    def characterReceived(self, keyID, modifier):
        if keyID == b'\r':
            self.onSubmit(self.buffer)
        elif len(self.buffer) < self.maxwidth:
            self.buffer = self.buffer[:self.cursor] + keyID + self.buffer[self.cursor:]
            self.cursor += 1
            self.repaint()

    def sizeHint(self):
        return (self.maxwidth + 1, 1)

    def render(self, width, height, terminal):
        currentText = self._renderText()
        terminal.cursorPosition(0, 0)
        if self.focused:
            terminal.write(currentText[:self.cursor])
            cursor(terminal, currentText[self.cursor:self.cursor + 1] or b' ')
            terminal.write(currentText[self.cursor + 1:])
            terminal.write(b' ' * (self.maxwidth - len(currentText) + 1))
        else:
            more = self.maxwidth - len(currentText)
            terminal.write(currentText + b'_' * more)

    def _renderText(self):
        return self.buffer