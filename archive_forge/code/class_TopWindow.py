import array
from twisted.conch.insults import helper, insults
from twisted.python import text as tptext
class TopWindow(ContainerWidget):
    """
    A top-level container object which provides focus wrap-around and paint
    scheduling.

    @ivar painter: A no-argument callable which will be invoked when this
    widget needs to be redrawn.

    @ivar scheduler: A one-argument callable which will be invoked with a
    no-argument callable and should arrange for it to invoked at some point in
    the near future.  The no-argument callable will cause this widget and all
    its children to be redrawn.  It is typically beneficial for the no-argument
    callable to be invoked at the end of handling for whatever event is
    currently active; for example, it might make sense to call it at the end of
    L{twisted.conch.insults.insults.ITerminalProtocol.keystrokeReceived}.
    Note, however, that since calls to this may also be made in response to no
    apparent event, arrangements should be made for the function to be called
    even if an event handler such as C{keystrokeReceived} is not on the call
    stack (eg, using
    L{reactor.callLater<twisted.internet.interfaces.IReactorTime.callLater>}
    with a short timeout).
    """
    focused = True

    def __init__(self, painter, scheduler):
        ContainerWidget.__init__(self)
        self.painter = painter
        self.scheduler = scheduler
    _paintCall = None

    def repaint(self):
        if self._paintCall is None:
            self._paintCall = object()
            self.scheduler(self._paint)
        ContainerWidget.repaint(self)

    def _paint(self):
        self._paintCall = None
        self.painter()

    def changeFocus(self):
        try:
            ContainerWidget.changeFocus(self)
        except YieldFocus:
            try:
                ContainerWidget.changeFocus(self)
            except YieldFocus:
                pass

    def keystrokeReceived(self, keyID, modifier):
        try:
            ContainerWidget.keystrokeReceived(self, keyID, modifier)
        except YieldFocus:
            self.changeFocus()