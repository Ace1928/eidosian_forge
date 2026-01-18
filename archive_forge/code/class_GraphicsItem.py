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
class GraphicsItem(object):
    """
    **Bases:** :class:`object`

    Abstract class providing useful methods to GraphicsObject and GraphicsWidget.
    (This is required because we cannot have multiple inheritance with QObject subclasses.)

    A note about Qt's GraphicsView framework:

    The GraphicsView system places a lot of emphasis on the notion that the graphics within the scene should be device independent--you should be able to take the same graphics and display them on screens of different resolutions, printers, export to SVG, etc. This is nice in principle, but causes me a lot of headache in practice. It means that I have to circumvent all the device-independent expectations any time I want to operate in pixel coordinates rather than arbitrary scene coordinates. A lot of the code in GraphicsItem is devoted to this task--keeping track of view widgets and device transforms, computing the size and shape of a pixel in local item coordinates, etc. Note that in item coordinates, a pixel does not have to be square or even rectangular, so just asking how to increase a bounding rect by 2px can be a rather complex task.
    """
    _pixelVectorGlobalCache = LRU(100)

    def __init__(self):
        if not hasattr(self, '_qtBaseClass'):
            for b in self.__class__.__bases__:
                if issubclass(b, QtWidgets.QGraphicsItem):
                    self.__class__._qtBaseClass = b
                    break
        if not hasattr(self, '_qtBaseClass'):
            raise Exception('Could not determine Qt base class for GraphicsItem: %s' % str(self))
        self._pixelVectorCache = [None, None]
        self._viewWidget = None
        self._viewBox = None
        self._connectedView = None
        self._exportOpts = False
        self._cachedView = None

    def getViewWidget(self):
        """
        Return the view widget for this item. 
        
        If the scene has multiple views, only the first view is returned.
        The return value is cached; clear the cached value with forgetViewWidget().
        If the view has been deleted by Qt, return None.
        """
        if self._viewWidget is None:
            scene = self.scene()
            if scene is None:
                return None
            views = scene.views()
            if len(views) < 1:
                return None
            self._viewWidget = weakref.ref(self.scene().views()[0])
        v = self._viewWidget()
        if v is not None and (not isQObjectAlive(v)):
            return None
        return v

    def forgetViewWidget(self):
        self._viewWidget = None

    def getViewBox(self):
        """
        Return the first ViewBox or GraphicsView which bounds this item's visible space.
        If this item is not contained within a ViewBox, then the GraphicsView is returned.
        If the item is contained inside nested ViewBoxes, then the inner-most ViewBox is returned.
        The result is cached; clear the cache with forgetViewBox()
        """
        if self._viewBox is None:
            p = self
            while True:
                try:
                    p = p.parentItem()
                except RuntimeError:
                    return None
                if p is None:
                    vb = self.getViewWidget()
                    if vb is None:
                        return None
                    else:
                        self._viewBox = weakref.ref(vb)
                        break
                if hasattr(p, 'implements') and p.implements('ViewBox'):
                    self._viewBox = weakref.ref(p)
                    break
        return self._viewBox()

    def forgetViewBox(self):
        self._viewBox = None

    def deviceTransform(self, viewportTransform=None):
        """
        Return the transform that converts local item coordinates to device coordinates (usually pixels).
        Extends deviceTransform to automatically determine the viewportTransform.
        """
        if viewportTransform is None:
            view = self.getViewWidget()
            if view is None:
                return None
            viewportTransform = view.viewportTransform()
        dt = self._qtBaseClass.deviceTransform(self, viewportTransform)
        if dt.determinant() == 0:
            return None
        else:
            return dt

    def viewTransform(self):
        """Return the transform that maps from local coordinates to the item's ViewBox coordinates
        If there is no ViewBox, return the scene transform.
        Returns None if the item does not have a view."""
        view = self.getViewBox()
        if view is None:
            return None
        if hasattr(view, 'implements') and view.implements('ViewBox'):
            return self.itemTransform(view.innerSceneItem())[0]
        else:
            return self.sceneTransform()

    def getBoundingParents(self):
        """Return a list of parents to this item that have child clipping enabled."""
        p = self
        parents = []
        while True:
            p = p.parentItem()
            if p is None:
                break
            if p.flags() & self.GraphicsItemFlag.ItemClipsChildrenToShape:
                parents.append(p)
        return parents

    def viewRect(self):
        """Return the visible bounds of this item's ViewBox or GraphicsWidget,
        in the local coordinate system of the item."""
        if self._cachedView is not None:
            return self._cachedView
        view = self.getViewBox()
        if view is None:
            return None
        bounds = self.mapRectFromView(view.viewRect())
        if bounds is None:
            return None
        bounds = bounds.normalized()
        self._cachedView = bounds
        return bounds

    def pixelVectors(self, direction=None):
        """Return vectors in local coordinates representing the width and height of a view pixel.
        If direction is specified, then return vectors parallel and orthogonal to it.
        
        Return (None, None) if pixel size is not yet defined (usually because the item has not yet been displayed)
        or if pixel size is below floating-point precision limit.
        """
        dt = self.deviceTransform()
        if dt is None:
            return (None, None)
        dt.setMatrix(dt.m11(), dt.m12(), 0, dt.m21(), dt.m22(), 0, 0, 0, 1)
        if direction is None:
            direction = QtCore.QPointF(1, 0)
        elif direction.manhattanLength() == 0:
            raise Exception('Cannot compute pixel length for 0-length vector.')
        key = (dt.m11(), dt.m21(), dt.m12(), dt.m22(), direction.x(), direction.y())
        if key == self._pixelVectorCache[0]:
            return tuple(map(Point, self._pixelVectorCache[1]))
        pv = self._pixelVectorGlobalCache.get(key, None)
        if pv is not None:
            self._pixelVectorCache = [key, pv]
            return tuple(map(Point, pv))
        directionr = direction
        dirLine = QtCore.QLineF(QtCore.QPointF(0, 0), directionr)
        viewDir = dt.map(dirLine)
        if viewDir.length() == 0:
            return (None, None)
        try:
            normView = viewDir.unitVector()
            normOrtho = normView.normalVector()
        except:
            raise Exception('Invalid direction %s' % directionr)
        dti = fn.invertQTransform(dt)
        pv = (Point(dti.map(normView).p2()), Point(dti.map(normOrtho).p2()))
        self._pixelVectorCache[1] = pv
        self._pixelVectorCache[0] = key
        self._pixelVectorGlobalCache[key] = pv
        return self._pixelVectorCache[1]

    def pixelLength(self, direction, ortho=False):
        """Return the length of one pixel in the direction indicated (in local coordinates)
        If ortho=True, then return the length of one pixel orthogonal to the direction indicated.
        
        Return None if pixel size is not yet defined (usually because the item has not yet been displayed).
        """
        normV, orthoV = self.pixelVectors(direction)
        if normV is None or orthoV is None:
            return None
        if ortho:
            return orthoV.length()
        return normV.length()

    def pixelSize(self):
        v = self.pixelVectors()
        if v == (None, None):
            return (None, None)
        return (hypot(v[0].x(), v[0].y()), hypot(v[1].x(), v[1].y()))

    def pixelWidth(self):
        vt = self.deviceTransform()
        if vt is None:
            return 0
        vt = fn.invertQTransform(vt)
        return vt.map(QtCore.QLineF(0, 0, 1, 0)).length()

    def pixelHeight(self):
        vt = self.deviceTransform()
        if vt is None:
            return 0
        vt = fn.invertQTransform(vt)
        return vt.map(QtCore.QLineF(0, 0, 0, 1)).length()

    def mapToDevice(self, obj):
        """
        Return *obj* mapped from local coordinates to device coordinates (pixels).
        If there is no device mapping available, return None.
        """
        vt = self.deviceTransform()
        if vt is None:
            return None
        return vt.map(obj)

    def mapFromDevice(self, obj):
        """
        Return *obj* mapped from device coordinates (pixels) to local coordinates.
        If there is no device mapping available, return None.
        """
        vt = self.deviceTransform()
        if vt is None:
            return None
        if isinstance(obj, QtCore.QPoint):
            obj = QtCore.QPointF(obj)
        vt = fn.invertQTransform(vt)
        return vt.map(obj)

    def mapRectToDevice(self, rect):
        """
        Return *rect* mapped from local coordinates to device coordinates (pixels).
        If there is no device mapping available, return None.
        """
        vt = self.deviceTransform()
        if vt is None:
            return None
        return vt.mapRect(rect)

    def mapRectFromDevice(self, rect):
        """
        Return *rect* mapped from device coordinates (pixels) to local coordinates.
        If there is no device mapping available, return None.
        """
        vt = self.deviceTransform()
        if vt is None:
            return None
        vt = fn.invertQTransform(vt)
        return vt.mapRect(rect)

    def mapToView(self, obj):
        vt = self.viewTransform()
        if vt is None:
            return None
        return vt.map(obj)

    def mapRectToView(self, obj):
        vt = self.viewTransform()
        if vt is None:
            return None
        return vt.mapRect(obj)

    def mapFromView(self, obj):
        vt = self.viewTransform()
        if vt is None:
            return None
        vt = fn.invertQTransform(vt)
        return vt.map(obj)

    def mapRectFromView(self, obj):
        vt = self.viewTransform()
        if vt is None:
            return None
        vt = fn.invertQTransform(vt)
        return vt.mapRect(obj)

    def pos(self):
        return Point(self._qtBaseClass.pos(self))

    def viewPos(self):
        return self.mapToView(self.mapFromParent(self.pos()))

    def parentItem(self):
        return self._qtBaseClass.parentItem(self)

    def setParentItem(self, parent):
        if parent is not None:
            pscene = parent.scene()
            if pscene is not None and self.scene() is not pscene:
                pscene.addItem(self)
        return self._qtBaseClass.setParentItem(self, parent)

    def childItems(self):
        return self._qtBaseClass.childItems(self)

    def sceneTransform(self):
        if self.scene() is None:
            return self.transform()
        else:
            return self._qtBaseClass.sceneTransform(self)

    def transformAngle(self, relativeItem=None):
        """Return the rotation produced by this item's transform (this assumes there is no shear in the transform)
        If relativeItem is given, then the angle is determined relative to that item.
        """
        if relativeItem is None:
            relativeItem = self.parentItem()
        tr = self.itemTransform(relativeItem)[0]
        vec = tr.map(QtCore.QLineF(0, 0, 1, 0))
        return vec.angleTo(QtCore.QLineF(vec.p1(), vec.p1() + QtCore.QPointF(1, 0)))

    def changeParent(self):
        """Called when the item's parent has changed. 
        This method handles connecting / disconnecting from ViewBox signals
        to make sure viewRangeChanged works properly. It should generally be 
        extended, not overridden."""
        self._updateView()

    def parentChanged(self):
        GraphicsItem.changeParent(self)

    def _updateView(self):
        if not hasattr(self, '_connectedView'):
            return
        self.forgetViewBox()
        self.forgetViewWidget()
        view = self.getViewBox()
        oldView = None
        if self._connectedView is not None:
            oldView = self._connectedView()
        if view is oldView:
            return
        if oldView is not None:
            Device = 'Device' if hasattr(oldView, 'sigDeviceRangeChanged') else ''
            for signal, slot in [(f'sig{Device}RangeChanged', self.viewRangeChanged), (f'sig{Device}TransformChanged', self.viewTransformChanged)]:
                try:
                    getattr(oldView, signal).disconnect(slot)
                except (TypeError, AttributeError, RuntimeError):
                    pass
            self._connectedView = None
        if view is not None:
            if hasattr(view, 'sigDeviceRangeChanged'):
                view.sigDeviceRangeChanged.connect(self.viewRangeChanged)
                view.sigDeviceTransformChanged.connect(self.viewTransformChanged)
            else:
                view.sigRangeChanged.connect(self.viewRangeChanged)
                view.sigTransformChanged.connect(self.viewTransformChanged)
            self._connectedView = weakref.ref(view)
            self.viewRangeChanged()
            self.viewTransformChanged()
        self._replaceView(oldView)
        self.viewChanged(view, oldView)

    def viewChanged(self, view, oldView):
        """Called when this item's view has changed
        (ie, the item has been added to or removed from a ViewBox)"""

    def _replaceView(self, oldView, item=None):
        if item is None:
            item = self
        for child in item.childItems():
            if isinstance(child, GraphicsItem):
                if child.getViewBox() is oldView:
                    child._updateView()
            else:
                self._replaceView(oldView, child)

    def viewRangeChanged(self):
        """
        Called whenever the view coordinates of the ViewBox containing this item have changed.
        """

    def viewTransformChanged(self):
        """
        Called whenever the transformation matrix of the view has changed.
        (eg, the view range has changed or the view was resized)
        Invalidates the viewRect cache.
        """
        self._cachedView = None

    def informViewBoundsChanged(self):
        """
        Inform this item's container ViewBox that the bounds of this item have changed.
        This is used by ViewBox to react if auto-range is enabled.
        """
        view = self.getViewBox()
        if view is not None and hasattr(view, 'implements') and view.implements('ViewBox'):
            view.itemBoundsChanged(self)

    def childrenShape(self):
        """Return the union of the shapes of all descendants of this item in local coordinates."""
        shapes = [self.mapFromItem(c, c.shape()) for c in self.allChildItems()]
        return reduce(operator.add, shapes)

    def allChildItems(self, root=None):
        """Return list of the entire item tree descending from this item."""
        if root is None:
            root = self
        tree = []
        for ch in root.childItems():
            tree.append(ch)
            tree.extend(self.allChildItems(ch))
        return tree

    def setExportMode(self, export, opts=None):
        """
        This method is called by exporters to inform items that they are being drawn for export
        with a specific set of options. Items access these via self._exportOptions.
        When exporting is complete, _exportOptions is set to False.
        """
        if opts is None:
            opts = {}
        if export:
            self._exportOpts = opts
        else:
            self._exportOpts = False

    def getContextMenus(self, event):
        return [self.getMenu()] if hasattr(self, 'getMenu') else []

    def generateSvg(self, nodes: dict[str, Element]) -> Optional[tuple[Element, list[Element]]]:
        """Method to override to manually specify the SVG writer mechanism.

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

        """
        raise NotImplementedError