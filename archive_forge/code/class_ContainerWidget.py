import array
from twisted.conch.insults import helper, insults
from twisted.python import text as tptext
class ContainerWidget(Widget):
    """
    @ivar focusedChild: The contained widget which currently has
    focus, or None.
    """
    focusedChild = None
    focused = False

    def __init__(self):
        Widget.__init__(self)
        self.children = []

    def addChild(self, child):
        assert child.parent is None
        child.parent = self
        self.children.append(child)
        if self.focusedChild is None and self.focused:
            try:
                child.focusReceived()
            except YieldFocus:
                pass
            else:
                self.focusedChild = child
        self.repaint()

    def remChild(self, child):
        assert child.parent is self
        child.parent = None
        self.children.remove(child)
        self.repaint()

    def filthy(self):
        for ch in self.children:
            ch.filthy()
        Widget.filthy(self)

    def render(self, width, height, terminal):
        for ch in self.children:
            ch.draw(width, height, terminal)

    def changeFocus(self):
        self.repaint()
        if self.focusedChild is not None:
            self.focusedChild.focusLost()
            focusedChild = self.focusedChild
            self.focusedChild = None
            try:
                curFocus = self.children.index(focusedChild) + 1
            except ValueError:
                raise YieldFocus()
        else:
            curFocus = 0
        while curFocus < len(self.children):
            try:
                self.children[curFocus].focusReceived()
            except YieldFocus:
                curFocus += 1
            else:
                self.focusedChild = self.children[curFocus]
                return
        raise YieldFocus()

    def focusReceived(self):
        self.changeFocus()
        self.focused = True

    def keystrokeReceived(self, keyID, modifier):
        if self.focusedChild is not None:
            try:
                self.focusedChild.keystrokeReceived(keyID, modifier)
            except YieldFocus:
                self.changeFocus()
                self.repaint()
        else:
            Widget.keystrokeReceived(self, keyID, modifier)