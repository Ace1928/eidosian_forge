import array
from twisted.conch.insults import helper, insults
from twisted.python import text as tptext
class HorizontalScrollbar(_Scrollbar):

    def sizeHint(self):
        return (None, 1)

    def func_LEFT_ARROW(self, modifier):
        self.smaller()

    def func_RIGHT_ARROW(self, modifier):
        self.bigger()
    _left = '◀'
    _right = '▶'
    _bar = '░'
    _slider = '▓'

    def render(self, width, height, terminal):
        terminal.cursorPosition(0, 0)
        n = width - 3
        before = int(n * self.percent)
        after = n - before
        me = self._left + self._bar * before + self._slider + self._bar * after + self._right
        terminal.write(me.encode('utf-8'))