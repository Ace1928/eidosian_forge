from collections import namedtuple
from contextlib import ExitStack, contextmanager, nullcontext
from enum import Enum, IntEnum
import functools
import importlib
import inspect
import io
import itertools
import logging
import os
import sys
import time
import weakref
from weakref import WeakKeyDictionary
import numpy as np
import matplotlib as mpl
from matplotlib import (
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_managers import ToolManager
from matplotlib.cbook import _setattr_cm
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.path import Path
from matplotlib.texmanager import TexManager
from matplotlib.transforms import Affine2D
from matplotlib._enums import JoinStyle, CapStyle
class ToolContainerBase:
    """
    Base class for all tool containers, e.g. toolbars.

    Attributes
    ----------
    toolmanager : `.ToolManager`
        The tools with which this `ToolContainer` wants to communicate.
    """
    _icon_extension = '.png'
    '\n    Toolcontainer button icon image format extension\n\n    **String**: Image extension\n    '

    def __init__(self, toolmanager):
        self.toolmanager = toolmanager
        toolmanager.toolmanager_connect('tool_message_event', lambda event: self.set_message(event.message))
        toolmanager.toolmanager_connect('tool_removed_event', lambda event: self.remove_toolitem(event.tool.name))

    def _tool_toggled_cbk(self, event):
        """
        Capture the 'tool_trigger_[name]'

        This only gets used for toggled tools.
        """
        self.toggle_toolitem(event.tool.name, event.tool.toggled)

    def add_tool(self, tool, group, position=-1):
        """
        Add a tool to this container.

        Parameters
        ----------
        tool : tool_like
            The tool to add, see `.ToolManager.get_tool`.
        group : str
            The name of the group to add this tool to.
        position : int, default: -1
            The position within the group to place this tool.
        """
        tool = self.toolmanager.get_tool(tool)
        image = self._get_image_filename(tool.image)
        toggle = getattr(tool, 'toggled', None) is not None
        self.add_toolitem(tool.name, group, position, image, tool.description, toggle)
        if toggle:
            self.toolmanager.toolmanager_connect('tool_trigger_%s' % tool.name, self._tool_toggled_cbk)
            if tool.toggled:
                self.toggle_toolitem(tool.name, True)

    def _get_image_filename(self, image):
        """Find the image based on its name."""
        if not image:
            return None
        basedir = cbook._get_data_path('images')
        for fname in [image, image + self._icon_extension, str(basedir / image), str(basedir / (image + self._icon_extension))]:
            if os.path.isfile(fname):
                return fname

    def trigger_tool(self, name):
        """
        Trigger the tool.

        Parameters
        ----------
        name : str
            Name (id) of the tool triggered from within the container.
        """
        self.toolmanager.trigger_tool(name, sender=self)

    def add_toolitem(self, name, group, position, image, description, toggle):
        """
        Add a toolitem to the container.

        This method must be implemented per backend.

        The callback associated with the button click event,
        must be *exactly* ``self.trigger_tool(name)``.

        Parameters
        ----------
        name : str
            Name of the tool to add, this gets used as the tool's ID and as the
            default label of the buttons.
        group : str
            Name of the group that this tool belongs to.
        position : int
            Position of the tool within its group, if -1 it goes at the end.
        image : str
            Filename of the image for the button or `None`.
        description : str
            Description of the tool, used for the tooltips.
        toggle : bool
            * `True` : The button is a toggle (change the pressed/unpressed
              state between consecutive clicks).
            * `False` : The button is a normal button (returns to unpressed
              state after release).
        """
        raise NotImplementedError

    def toggle_toolitem(self, name, toggled):
        """
        Toggle the toolitem without firing event.

        Parameters
        ----------
        name : str
            Id of the tool to toggle.
        toggled : bool
            Whether to set this tool as toggled or not.
        """
        raise NotImplementedError

    def remove_toolitem(self, name):
        """
        Remove a toolitem from the `ToolContainer`.

        This method must get implemented per backend.

        Called when `.ToolManager` emits a `tool_removed_event`.

        Parameters
        ----------
        name : str
            Name of the tool to remove.
        """
        raise NotImplementedError

    def set_message(self, s):
        """
        Display a message on the toolbar.

        Parameters
        ----------
        s : str
            Message text.
        """
        raise NotImplementedError