from abc import ABCMeta, abstractmethod
from tkinter import (
from tkinter.filedialog import asksaveasfilename
from nltk.util import in_idle
def _add_child_widget(self, child):
    """
        Register a hierarchical child widget.  The child will be
        considered part of this canvas widget for purposes of user
        interaction.  ``_add_child_widget`` has two direct effects:
          - It sets ``child``'s parent to this canvas widget.
          - It adds ``child`` to the list of canvas widgets returned by
            the ``child_widgets`` member function.

        :param child: The new child widget.  ``child`` must not already
            have a parent.
        :type child: CanvasWidget
        """
    if not hasattr(self, '_CanvasWidget__children'):
        self.__children = []
    if child.__parent is not None:
        raise ValueError(f'{child} already has a parent')
    child.__parent = self
    self.__children.append(child)