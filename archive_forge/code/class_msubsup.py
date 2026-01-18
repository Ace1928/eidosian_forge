import docutils.utils.math.tex2unichar as tex2unichar
class msubsup(math):
    nchildren = 3

    def __init__(self, children=None, reversed=False):
        self.reversed = reversed
        math.__init__(self, children)

    def xml(self):
        if self.reversed:
            self.children[1:3] = [self.children[2], self.children[1]]
            self.reversed = False
        return math.xml(self)