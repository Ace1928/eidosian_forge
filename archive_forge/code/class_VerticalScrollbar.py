import array
from twisted.conch.insults import helper, insults
from twisted.python import text as tptext
class VerticalScrollbar(_Scrollbar):

    def sizeHint(self):
        return (1, None)

    def func_UP_ARROW(self, modifier):
        self.smaller()

    def func_DOWN_ARROW(self, modifier):
        self.bigger()
    _up = '▲'
    _down = '▼'
    _bar = '░'
    _slider = '▓'

    def render(self, width, height, terminal):
        terminal.cursorPosition(0, 0)
        knob = int(self.percent * (height - 2))
        terminal.write(self._up.encode('utf-8'))
        for i in range(1, height - 1):
            terminal.cursorPosition(0, i)
            if i != knob + 1:
                terminal.write(self._bar.encode('utf-8'))
            else:
                terminal.write(self._slider.encode('utf-8'))
        terminal.cursorPosition(0, height - 1)
        terminal.write(self._down.encode('utf-8'))