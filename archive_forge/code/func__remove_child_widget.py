from abc import ABCMeta, abstractmethod
from tkinter import (
from tkinter.filedialog import asksaveasfilename
from nltk.util import in_idle
def _remove_child_widget(self, child):
    """
        Remove a hierarchical child widget.  This child will no longer
        be considered part of this canvas widget for purposes of user
        interaction.  ``_add_child_widget`` has two direct effects:
          - It sets ``child``'s parent to None.
          - It removes ``child`` from the list of canvas widgets
            returned by the ``child_widgets`` member function.

        :param child: The child widget to remove.  ``child`` must be a
            child of this canvas widget.
        :type child: CanvasWidget
        """
    self.__children.remove(child)
    child.__parent = None