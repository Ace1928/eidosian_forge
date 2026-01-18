import array
from twisted.conch.insults import helper, insults
from twisted.python import text as tptext
class AbsoluteBox(ContainerWidget):

    def moveChild(self, child, x, y):
        for n in range(len(self.children)):
            if self.children[n][0] is child:
                self.children[n] = (child, x, y)
                break
        else:
            raise ValueError('No such child', child)

    def render(self, width, height, terminal):
        for ch, x, y in self.children:
            wrap = BoundedTerminalWrapper(terminal, width - x, height - y, x, y)
            ch.draw(width, height, wrap)