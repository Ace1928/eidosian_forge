import array
from twisted.conch.insults import helper, insults
from twisted.python import text as tptext
class BoundedTerminalWrapper:

    def __init__(self, terminal, width, height, xoff, yoff):
        self.width = width
        self.height = height
        self.xoff = xoff
        self.yoff = yoff
        self.terminal = terminal
        self.cursorForward = terminal.cursorForward
        self.selectCharacterSet = terminal.selectCharacterSet
        self.selectGraphicRendition = terminal.selectGraphicRendition
        self.saveCursor = terminal.saveCursor
        self.restoreCursor = terminal.restoreCursor

    def cursorPosition(self, x, y):
        return self.terminal.cursorPosition(self.xoff + min(self.width, x), self.yoff + min(self.height, y))

    def cursorHome(self):
        return self.terminal.cursorPosition(self.xoff, self.yoff)

    def write(self, data):
        return self.terminal.write(data)