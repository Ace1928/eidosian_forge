import itertools
import math
import weakref
from collections import OrderedDict
import numpy as np
from .. import Qt, debug
from .. import functions as fn
from .. import getConfigOption
from ..Point import Point
from ..Qt import QtCore, QtGui
from .GraphicsObject import GraphicsObject
class ScatterPlotItem(GraphicsObject):
    """
    Displays a set of x/y points. Instances of this class are created
    automatically as part of PlotDataItem; these rarely need to be instantiated
    directly.

    The size, shape, pen, and fill brush may be set for each point individually
    or for all points.


    ============================  ===============================================
    **Signals:**
    sigPlotChanged(self)          Emitted when the data being plotted has changed
    sigClicked(self, points, ev)  Emitted when points are clicked. Sends a list
                                  of all the points under the mouse pointer.
    sigHovered(self, points, ev)  Emitted when the item is hovered. Sends a list
                                  of all the points under the mouse pointer.
    ============================  ===============================================

    """
    sigClicked = QtCore.Signal(object, object, object)
    sigHovered = QtCore.Signal(object, object, object)
    sigPlotChanged = QtCore.Signal(object)

    def __init__(self, *args, **kargs):
        """
        Accepts the same arguments as setData()
        """
        profiler = debug.Profiler()
        GraphicsObject.__init__(self)
        self.picture = None
        self.fragmentAtlas = SymbolAtlas()
        dtype = [('x', float), ('y', float), ('size', float), ('symbol', object), ('pen', object), ('brush', object), ('visible', bool), ('data', object), ('hovered', bool), ('item', object), ('sourceRect', [('x', int), ('y', int), ('w', int), ('h', int)])]
        self.data = np.empty(0, dtype=dtype)
        self.bounds = [None, None]
        self._maxSpotWidth = 0
        self._maxSpotPxWidth = 0
        self._pixmapFragments = Qt.internals.PrimitiveArray(QtGui.QPainter.PixmapFragment, 10)
        self.opts = {'pxMode': True, 'useCache': True, 'antialias': getConfigOption('antialias'), 'compositionMode': None, 'name': None, 'symbol': 'o', 'size': 7, 'pen': fn.mkPen(getConfigOption('foreground')), 'brush': fn.mkBrush(100, 100, 150), 'hoverable': False, 'tip': 'x: {x:.3g}\ny: {y:.3g}\ndata={data}'.format}
        self.opts.update({'hover' + opt.title(): _DEFAULT_STYLE[opt] for opt in ['symbol', 'size', 'pen', 'brush']})
        profiler()
        self.setData(*args, **kargs)
        profiler('setData')
        self._toolTipCleared = True

    def setData(self, *args, **kargs):
        """
        **Ordered Arguments:**

        * If there is only one unnamed argument, it will be interpreted like the 'spots' argument.
        * If there are two unnamed arguments, they will be interpreted as sequences of x and y values.

        ====================== ===============================================================================================
        **Keyword Arguments:**
        *spots*                Optional list of dicts. Each dict specifies parameters for a single spot:
                               {'pos': (x,y), 'size', 'pen', 'brush', 'symbol'}. This is just an alternate method
                               of passing in data for the corresponding arguments.
        *x*,*y*                1D arrays of x,y values.
        *pos*                  2D structure of x,y pairs (such as Nx2 array or list of tuples)
        *pxMode*               If True, spots are always the same size regardless of scaling, and size is given in px.
                               Otherwise, size is in scene coordinates and the spots scale with the view. To ensure
                               effective caching, QPen and QBrush objects should be reused as much as possible.
                               Default is True
        *symbol*               can be one (or a list) of symbols. For a list of supported symbols, see 
                               :func:`~ScatterPlotItem.setSymbol`. QPainterPath is also supported to specify custom symbol
                               shapes. To properly obey the position and size, custom symbols should be centered at (0,0) and
                               width and height of 1.0. Note that it is also possible to 'install' custom shapes by setting 
                               ScatterPlotItem.Symbols[key] = shape.
        *pen*                  The pen (or list of pens) to use for drawing spot outlines.
        *brush*                The brush (or list of brushes) to use for filling spots.
        *size*                 The size (or list of sizes) of spots. If *pxMode* is True, this value is in pixels. Otherwise,
                               it is in the item's local coordinate system.
        *data*                 a list of python objects used to uniquely identify each spot.
        *hoverable*            If True, sigHovered is emitted with a list of hovered points, a tool tip is shown containing
                               information about them, and an optional separate style for them is used. Default is False.
        *tip*                  A string-valued function of a spot's (x, y, data) values. Set to None to prevent a tool tip
                               from being shown.
        *hoverSymbol*          A single symbol to use for hovered spots. Set to None to keep symbol unchanged. Default is None.
        *hoverSize*            A single size to use for hovered spots. Set to -1 to keep size unchanged. Default is -1.
        *hoverPen*             A single pen to use for hovered spots. Set to None to keep pen unchanged. Default is None.
        *hoverBrush*           A single brush to use for hovered spots. Set to None to keep brush unchanged. Default is None.
        *useCache*             (bool) By default, generated point graphics items are cached to
                               improve performance. Setting this to False can improve image quality
                               in certain situations.
        *antialias*            Whether to draw symbols with antialiasing. Note that if pxMode is True, symbols are
                               always rendered with antialiasing (since the rendered symbols can be cached, this
                               incurs very little performance cost)
        *compositionMode*      If specified, this sets the composition mode used when drawing the
                               scatter plot (see QPainter::CompositionMode in the Qt documentation).
        *name*                 The name of this item. Names are used for automatically
                               generating LegendItem entries and by some exporters.
        ====================== ===============================================================================================
        """
        oldData = self.data
        self.clear()
        self.addPoints(*args, **kargs)

    def addPoints(self, *args, **kargs):
        """
        Add new points to the scatter plot.
        Arguments are the same as setData()
        """
        if len(args) == 1:
            kargs['spots'] = args[0]
        elif len(args) == 2:
            kargs['x'] = args[0]
            kargs['y'] = args[1]
        elif len(args) > 2:
            raise Exception('Only accepts up to two non-keyword arguments.')
        if 'pos' in kargs:
            pos = kargs['pos']
            if isinstance(pos, np.ndarray):
                kargs['x'] = pos[:, 0]
                kargs['y'] = pos[:, 1]
            else:
                x = []
                y = []
                for p in pos:
                    if isinstance(p, QtCore.QPointF):
                        x.append(p.x())
                        y.append(p.y())
                    else:
                        x.append(p[0])
                        y.append(p[1])
                kargs['x'] = x
                kargs['y'] = y
        if 'spots' in kargs:
            numPts = len(kargs['spots'])
        elif 'y' in kargs and kargs['y'] is not None:
            numPts = len(kargs['y'])
        else:
            kargs['x'] = []
            kargs['y'] = []
            numPts = 0
        self.data['item'][...] = None
        oldData = self.data
        self.data = np.empty(len(oldData) + numPts, dtype=self.data.dtype)
        self.data[:len(oldData)] = oldData
        newData = self.data[len(oldData):]
        newData['size'] = -1
        newData['visible'] = True
        if 'spots' in kargs:
            spots = kargs['spots']
            for i in range(len(spots)):
                spot = spots[i]
                for k in spot:
                    if k == 'pos':
                        pos = spot[k]
                        if isinstance(pos, QtCore.QPointF):
                            x, y = (pos.x(), pos.y())
                        else:
                            x, y = (pos[0], pos[1])
                        newData[i]['x'] = x
                        newData[i]['y'] = y
                    elif k == 'pen':
                        newData[i][k] = _mkPen(spot[k])
                    elif k == 'brush':
                        newData[i][k] = _mkBrush(spot[k])
                    elif k in ['x', 'y', 'size', 'symbol', 'data']:
                        newData[i][k] = spot[k]
                    else:
                        raise Exception('Unknown spot parameter: %s' % k)
        elif 'y' in kargs:
            newData['x'] = kargs['x']
            newData['y'] = kargs['y']
        if 'name' in kargs:
            self.opts['name'] = kargs['name']
        if 'pxMode' in kargs:
            self.setPxMode(kargs['pxMode'])
        if 'antialias' in kargs:
            self.opts['antialias'] = kargs['antialias']
        if 'hoverable' in kargs:
            self.opts['hoverable'] = bool(kargs['hoverable'])
        if 'tip' in kargs:
            self.opts['tip'] = kargs['tip']
        if 'useCache' in kargs:
            self.opts['useCache'] = kargs['useCache']
        for k in ['pen', 'brush', 'symbol', 'size']:
            if k in kargs:
                setMethod = getattr(self, 'set' + k[0].upper() + k[1:])
                setMethod(kargs[k], update=False, dataSet=newData, mask=kargs.get('mask', None))
            kh = 'hover' + k.title()
            if kh in kargs:
                vh = kargs[kh]
                if k == 'pen':
                    vh = _mkPen(vh)
                elif k == 'brush':
                    vh = _mkBrush(vh)
                self.opts[kh] = vh
        if 'data' in kargs:
            self.setPointData(kargs['data'], dataSet=newData)
        self.prepareGeometryChange()
        self.informViewBoundsChanged()
        self.bounds = [None, None]
        self.invalidate()
        self.updateSpots(newData)
        self.sigPlotChanged.emit(self)

    def invalidate(self):
        self.picture = None
        self.update()

    def getData(self):
        return (self.data['x'], self.data['y'])

    def implements(self, interface=None):
        ints = ['plotData']
        if interface is None:
            return ints
        return interface in ints

    def name(self):
        return self.opts.get('name', None)

    def setPen(self, *args, **kargs):
        """Set the pen(s) used to draw the outline around each spot.
        If a list or array is provided, then the pen for each spot will be set separately.
        Otherwise, the arguments are passed to pg.mkPen and used as the default pen for
        all spots which do not have a pen explicitly set."""
        update = kargs.pop('update', True)
        dataSet = kargs.pop('dataSet', self.data)
        if len(args) == 1 and (isinstance(args[0], np.ndarray) or isinstance(args[0], list)):
            pens = args[0]
            if 'mask' in kargs and kargs['mask'] is not None:
                pens = pens[kargs['mask']]
            if len(pens) != len(dataSet):
                raise Exception('Number of pens does not match number of points (%d != %d)' % (len(pens), len(dataSet)))
            dataSet['pen'] = list(map(_mkPen, pens))
        else:
            self.opts['pen'] = _mkPen(*args, **kargs)
        dataSet['sourceRect'] = 0
        if update:
            self.updateSpots(dataSet)

    def setBrush(self, *args, **kargs):
        """Set the brush(es) used to fill the interior of each spot.
        If a list or array is provided, then the brush for each spot will be set separately.
        Otherwise, the arguments are passed to pg.mkBrush and used as the default brush for
        all spots which do not have a brush explicitly set."""
        update = kargs.pop('update', True)
        dataSet = kargs.pop('dataSet', self.data)
        if len(args) == 1 and (isinstance(args[0], np.ndarray) or isinstance(args[0], list)):
            brushes = args[0]
            if 'mask' in kargs and kargs['mask'] is not None:
                brushes = brushes[kargs['mask']]
            if len(brushes) != len(dataSet):
                raise Exception('Number of brushes does not match number of points (%d != %d)' % (len(brushes), len(dataSet)))
            dataSet['brush'] = list(map(_mkBrush, brushes))
        else:
            self.opts['brush'] = _mkBrush(*args, **kargs)
        dataSet['sourceRect'] = 0
        if update:
            self.updateSpots(dataSet)

    def setSymbol(self, symbol, update=True, dataSet=None, mask=None):
        """Set the symbol(s) used to draw each spot.
        If a list or array is provided, then the symbol for each spot will be set separately.
        Otherwise, the argument will be used as the default symbol for
        all spots which do not have a symbol explicitly set.

        **Supported symbols:**

        * 'o'  circle (default)
        * 's'  square
        * 't'  triangle
        * 'd'  diamond
        * '+'  plus
        * 't1' triangle pointing upwards
        * 't2'  triangle pointing right side
        * 't3'  triangle pointing left side
        * 'p'  pentagon
        * 'h'  hexagon
        * 'star'
        * 'x'  cross
        * 'arrow_up'
        * 'arrow_right'
        * 'arrow_down'
        * 'arrow_left'
        * 'crosshair'
        * any QPainterPath to specify custom symbol shapes.

        """
        if dataSet is None:
            dataSet = self.data
        if isinstance(symbol, np.ndarray) or isinstance(symbol, list):
            symbols = symbol
            if mask is not None:
                symbols = symbols[mask]
            if len(symbols) != len(dataSet):
                raise Exception('Number of symbols does not match number of points (%d != %d)' % (len(symbols), len(dataSet)))
            dataSet['symbol'] = symbols
        else:
            self.opts['symbol'] = symbol
            self._spotPixmap = None
        dataSet['sourceRect'] = 0
        if update:
            self.updateSpots(dataSet)

    def setSize(self, size, update=True, dataSet=None, mask=None):
        """Set the size(s) used to draw each spot.
        If a list or array is provided, then the size for each spot will be set separately.
        Otherwise, the argument will be used as the default size for
        all spots which do not have a size explicitly set."""
        if dataSet is None:
            dataSet = self.data
        if isinstance(size, np.ndarray) or isinstance(size, list):
            sizes = size
            if mask is not None:
                sizes = sizes[mask]
            if len(sizes) != len(dataSet):
                raise Exception('Number of sizes does not match number of points (%d != %d)' % (len(sizes), len(dataSet)))
            dataSet['size'] = sizes
        else:
            self.opts['size'] = size
            self._spotPixmap = None
        dataSet['sourceRect'] = 0
        if update:
            self.updateSpots(dataSet)

    def setPointsVisible(self, visible, update=True, dataSet=None, mask=None):
        """Set whether or not each spot is visible.
        If a list or array is provided, then the visibility for each spot will be set separately.
        Otherwise, the argument will be used for all spots."""
        if dataSet is None:
            dataSet = self.data
        if isinstance(visible, np.ndarray) or isinstance(visible, list):
            visibilities = visible
            if mask is not None:
                visibilities = visibilities[mask]
            if len(visibilities) != len(dataSet):
                raise Exception('Number of visibilities does not match number of points (%d != %d)' % (len(visibilities), len(dataSet)))
            dataSet['visible'] = visibilities
        else:
            dataSet['visible'] = visible
        dataSet['sourceRect'] = 0
        if update:
            self.updateSpots(dataSet)

    def setPointData(self, data, dataSet=None, mask=None):
        if dataSet is None:
            dataSet = self.data
        if isinstance(data, np.ndarray) or isinstance(data, list):
            if mask is not None:
                data = data[mask]
            if len(data) != len(dataSet):
                raise Exception('Length of meta data does not match number of points (%d != %d)' % (len(data), len(dataSet)))
        if isinstance(data, np.ndarray) and data.dtype.fields is not None and (len(data.dtype.fields) > 1):
            for i, rec in enumerate(data):
                dataSet['data'][i] = rec
        else:
            dataSet['data'] = data

    def setPxMode(self, mode):
        if self.opts['pxMode'] == mode:
            return
        self.opts['pxMode'] = mode
        self.invalidate()

    def updateSpots(self, dataSet=None):
        profiler = debug.Profiler()
        if dataSet is None:
            dataSet = self.data
        invalidate = False
        if self.opts['pxMode'] and self.opts['useCache']:
            mask = dataSet['sourceRect']['w'] == 0
            if np.any(mask):
                invalidate = True
                coords = self.fragmentAtlas[list(zip(*self._style(['symbol', 'size', 'pen', 'brush'], data=dataSet, idx=mask)))]
                dataSet['sourceRect'][mask] = coords
            self._maybeRebuildAtlas()
        else:
            invalidate = True
        self._updateMaxSpotSizes(data=dataSet)
        if invalidate:
            self.invalidate()

    def _maybeRebuildAtlas(self, threshold=4, minlen=1000):
        n = len(self.fragmentAtlas)
        if n > minlen and n > threshold * len(self.data):
            self.fragmentAtlas.rebuild(list(zip(*self._style(['symbol', 'size', 'pen', 'brush']))))
            self.data['sourceRect'] = 0
            self.updateSpots()

    def _style(self, opts, data=None, idx=None, scale=None):
        if data is None:
            data = self.data
        if idx is None:
            idx = np.s_[:]
        for opt in opts:
            col = data[opt][idx]
            if col.base is not None:
                col = col.copy()
            if self.opts['hoverable']:
                val = self.opts['hover' + opt.title()]
                if val != _DEFAULT_STYLE[opt]:
                    col[data['hovered'][idx]] = val
            col[np.equal(col, _DEFAULT_STYLE[opt])] = self.opts[opt]
            if opt == 'size' and scale is not None:
                col *= scale
            yield col

    def _updateMaxSpotSizes(self, **kwargs):
        if self.opts['pxMode'] and self.opts['useCache']:
            w, pw = (0, self.fragmentAtlas.maxWidth)
        else:
            w, pw = max(itertools.chain([(self._maxSpotWidth, self._maxSpotPxWidth)], self._measureSpotSizes(**kwargs)))
        self._maxSpotWidth = w
        self._maxSpotPxWidth = pw
        self.bounds = [None, None]

    def _measureSpotSizes(self, **kwargs):
        """Generate pairs (width, pxWidth) for spots in data"""
        styles = zip(*self._style(['size', 'pen'], **kwargs))
        if self.opts['pxMode']:
            for size, pen in styles:
                yield (0, size + pen.widthF())
        else:
            for size, pen in styles:
                if pen.isCosmetic():
                    yield (size, pen.widthF())
                else:
                    yield (size + pen.widthF(), 0)

    def clear(self):
        """Remove all spots from the scatter plot"""
        self._maxSpotWidth = 0
        self._maxSpotPxWidth = 0
        self.data = np.empty(0, dtype=self.data.dtype)
        self.bounds = [None, None]
        self.invalidate()

    def dataBounds(self, ax, frac=1.0, orthoRange=None):
        if frac >= 1.0 and orthoRange is None and (self.bounds[ax] is not None):
            return self.bounds[ax]
        if self.data is None or len(self.data) == 0:
            return (None, None)
        if ax == 0:
            d = self.data['x']
            d2 = self.data['y']
        elif ax == 1:
            d = self.data['y']
            d2 = self.data['x']
        else:
            raise ValueError('Invalid axis value')
        if orthoRange is not None:
            mask = (d2 >= orthoRange[0]) * (d2 <= orthoRange[1])
            d = d[mask]
            if d.size == 0:
                return (None, None)
        if frac >= 1.0:
            self.bounds[ax] = (np.nanmin(d) - self._maxSpotWidth * 0.7072, np.nanmax(d) + self._maxSpotWidth * 0.7072)
            return self.bounds[ax]
        elif frac <= 0.0:
            raise Exception("Value for parameter 'frac' must be > 0. (got %s)" % str(frac))
        else:
            mask = np.isfinite(d)
            d = d[mask]
            return np.percentile(d, [50 * (1 - frac), 50 * (1 + frac)])

    def pixelPadding(self):
        return self._maxSpotPxWidth * 0.7072

    def boundingRect(self):
        xmn, xmx = self.dataBounds(ax=0)
        ymn, ymx = self.dataBounds(ax=1)
        if xmn is None or xmx is None:
            xmn = 0
            xmx = 0
        if ymn is None or ymx is None:
            ymn = 0
            ymx = 0
        px = py = 0.0
        pxPad = self.pixelPadding()
        if pxPad > 0:
            px, py = self.pixelVectors()
            try:
                px = 0 if px is None else px.length()
            except OverflowError:
                px = 0
            try:
                py = 0 if py is None else py.length()
            except OverflowError:
                py = 0
            px *= pxPad
            py *= pxPad
        return QtCore.QRectF(xmn - px, ymn - py, 2 * px + xmx - xmn, 2 * py + ymx - ymn)

    def viewTransformChanged(self):
        self.prepareGeometryChange()
        GraphicsObject.viewTransformChanged(self)
        self.bounds = [None, None]

    def setExportMode(self, *args, **kwds):
        GraphicsObject.setExportMode(self, *args, **kwds)
        self.invalidate()

    @debug.warnOnException
    def paint(self, p, option, widget):
        profiler = debug.Profiler()
        cmode = self.opts.get('compositionMode', None)
        if cmode is not None:
            p.setCompositionMode(cmode)
        if self._exportOpts is not False:
            aa = self._exportOpts.get('antialias', True)
            scale = self._exportOpts.get('resolutionScale', 1.0)
        else:
            aa = self.opts['antialias']
            scale = 1.0
        if self.opts['pxMode'] is True:
            viewMask = self._maskAt(self.viewRect())
            pts = np.vstack([self.data['x'], self.data['y']])
            pts = fn.transformCoordinates(p.transform(), pts)
            pts = fn.clip_array(pts, -2 ** 30, 2 ** 30)
            p.resetTransform()
            if self.opts['useCache'] and self._exportOpts is False:
                dpr = self.fragmentAtlas.devicePixelRatio()
                if widget is not None and (dpr_new := widget.devicePixelRatioF()) != dpr:
                    dpr = dpr_new
                    self.fragmentAtlas.setDevicePixelRatio(dpr)
                    self.fragmentAtlas.clear()
                    self.data['sourceRect'] = 0
                    self.updateSpots()
                xy = pts[:, viewMask].T
                sr = self.data['sourceRect'][viewMask]
                self._pixmapFragments.resize(sr.size)
                frags = self._pixmapFragments.ndarray()
                frags[:, 0:2] = xy
                frags[:, 2:6] = np.frombuffer(sr, dtype=int).reshape((-1, 4))
                frags[:, 6:10] = [1 / dpr, 1 / dpr, 0.0, 1.0]
                profiler('prep')
                drawargs = self._pixmapFragments.drawargs()
                p.drawPixmapFragments(*drawargs, self.fragmentAtlas.pixmap)
                profiler('draw')
            else:
                p.setRenderHint(p.RenderHint.Antialiasing, aa)
                for pt, style in zip(pts[:, viewMask].T, zip(*self._style(['symbol', 'size', 'pen', 'brush'], idx=viewMask, scale=scale))):
                    p.resetTransform()
                    p.translate(*pt)
                    drawSymbol(p, *style)
        else:
            if self.picture is None:
                self.picture = QtGui.QPicture()
                p2 = QtGui.QPainter(self.picture)
                for x, y, style in zip(self.data['x'], self.data['y'], zip(*self._style(['symbol', 'size', 'pen', 'brush'], scale=scale))):
                    p2.resetTransform()
                    p2.translate(x, y)
                    drawSymbol(p2, *style)
                p2.end()
            p.setRenderHint(p.RenderHint.Antialiasing, aa)
            self.picture.play(p)

    def points(self):
        m = np.equal(self.data['item'], None)
        for i in np.argwhere(m)[:, 0]:
            rec = self.data[i]
            if rec['item'] is None:
                rec['item'] = SpotItem(rec, self, i)
        return self.data['item']

    def pointsAt(self, pos):
        return self.points()[self._maskAt(pos)][::-1]

    def _maskAt(self, obj):
        """
        Return a boolean mask indicating all points that overlap obj, a QPointF or QRectF.
        """
        if isinstance(obj, QtCore.QPointF):
            l = r = obj.x()
            t = b = obj.y()
        elif isinstance(obj, QtCore.QRectF):
            l = obj.left()
            r = obj.right()
            t = obj.top()
            b = obj.bottom()
        else:
            raise TypeError
        if self.opts['pxMode'] and self.opts['useCache']:
            w = self.data['sourceRect']['w']
            h = self.data['sourceRect']['h']
        else:
            s, = self._style(['size'])
            w = h = s
        w = w / 2
        h = h / 2
        if self.opts['pxMode']:
            px, py = self.pixelVectors()
            try:
                px = 0 if px is None else px.length()
            except OverflowError:
                px = 0
            try:
                py = 0 if py is None else py.length()
            except OverflowError:
                py = 0
            w *= px
            h *= py
        return self.data['visible'] & (self.data['x'] + w > l) & (self.data['x'] - w < r) & (self.data['y'] + h > t) & (self.data['y'] - h < b)

    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            pts = self.pointsAt(ev.pos())
            if len(pts) > 0:
                self.ptsClicked = pts
                ev.accept()
                self.sigClicked.emit(self, self.ptsClicked, ev)
            else:
                ev.ignore()
        else:
            ev.ignore()

    def hoverEvent(self, ev):
        if self.opts['hoverable']:
            old = self.data['hovered']
            if ev.exit:
                new = np.zeros_like(self.data['hovered'])
            else:
                new = self._maskAt(ev.pos())
            if self._hasHoverStyle():
                self.data['sourceRect'][old ^ new] = 0
                self.data['hovered'] = new
                self.updateSpots()
            points = self.points()[new][::-1]
            vb = self.getViewBox()
            if vb is not None and self.opts['tip'] is not None:
                if len(points) > 0:
                    cutoff = 3
                    tip = [self.opts['tip'](x=pt.pos().x(), y=pt.pos().y(), data=pt.data()) for pt in points[:cutoff]]
                    if len(points) > cutoff:
                        tip.append('({} others...)'.format(len(points) - cutoff))
                    vb.setToolTip('\n\n'.join(tip))
                    self._toolTipCleared = False
                elif not self._toolTipCleared:
                    vb.setToolTip('')
                    self._toolTipCleared = True
            self.sigHovered.emit(self, points, ev)

    def _hasHoverStyle(self):
        return any((self.opts['hover' + opt.title()] != _DEFAULT_STYLE[opt] for opt in ['symbol', 'size', 'pen', 'brush']))