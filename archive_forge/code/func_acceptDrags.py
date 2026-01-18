import weakref
from time import perf_counter
from ..Point import Point
from ..Qt import QtCore
def acceptDrags(self, button):
    """Inform the scene that the item (that the event was delivered to)
        would accept a mouse drag event if the user were to drag before
        the next hover event.
        
        Returns True if the request is successful, otherwise returns False (indicating
        that some other item would receive an incoming drag event).
        """
    if not self.acceptable:
        return False
    if button not in self.__dragItems:
        self.__dragItems[button] = self.currentItem
        return True
    return False