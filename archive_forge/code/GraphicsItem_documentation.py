import operator
import weakref
from collections import OrderedDict
from functools import reduce
from math import hypot
from typing import Optional
from xml.etree.ElementTree import Element
from .. import functions as fn
from ..GraphicsScene import GraphicsScene
from ..Point import Point
from ..Qt import QtCore, QtWidgets, isQObjectAlive
Method to override to manually specify the SVG writer mechanism.

        Parameters
        ----------
        nodes
            Dictionary keyed by the name of graphics items and the XML
            representation of the the item that can be written as valid
            SVG.
        
        Returns
        -------
        tuple
            First element is the top level group for this item. The
            second element is a list of xml Elements corresponding to the
            child nodes of the item.
        None
            Return None if no XML is needed for rendering

        Raises
        ------
        NotImplementedError
            override method to implement in subclasses of GraphicsItem

        See Also
        --------
        pyqtgraph.exporters.SVGExporter._generateItemSvg
            The generic and default implementation

        