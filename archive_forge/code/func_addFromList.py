import logging
from reportlab import rl_config
def addFromList(self, drawlist, canv):
    """Consumes objects from the front of the list until the
        frame is full.  If it cannot fit one object, raises
        an exception."""
    if self._debug:
        logger.debug('enter Frame.addFromlist() for frame %s' % self.id)
    if self.showBoundary:
        self.drawBoundary(canv)
    while len(drawlist) > 0:
        head = drawlist[0]
        if self.add(head, canv, trySplit=0):
            del drawlist[0]
        else:
            break