import docutils.utils.math.tex2unichar as tex2unichar
class mover(math):
    nchildren = 2

    def __init__(self, children=None, reversed=False):
        self.reversed = reversed
        math.__init__(self, children)

    def xml(self):
        if self.reversed:
            self.children.reverse()
            self.reversed = False
        return math.xml(self)