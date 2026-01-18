from time import mktime, gmtime, strftime
from math import log10, pi, floor, sin, cos, hypot
import weakref
from reportlab.graphics.shapes import transformPoints, inverse, Ellipse, Group, String, numericXShift
from reportlab.lib.utils import flatten
from reportlab.pdfbase.pdfmetrics import stringWidth
class DrawTimeCollector:
    """
    generic mechanism for collecting information about nodes at the time they are about to be drawn
    """

    def __init__(self, formats=['gif']):
        self._nodes = weakref.WeakKeyDictionary()
        self.clear()
        self._pmcanv = None
        self.formats = formats
        self.disabled = False

    def clear(self):
        self._info = []
        self._info_append = self._info.append

    def record(self, func, node, *args, **kwds):
        self._nodes[node] = (func, args, kwds)
        node.__dict__['_drawTimeCallback'] = self

    def __call__(self, node, canvas, renderer):
        func = self._nodes.get(node, None)
        if func:
            func, args, kwds = func
            i = func(node, canvas, renderer, *args, **kwds)
            if i is not None:
                self._info_append(i)

    @staticmethod
    def rectDrawTimeCallback(node, canvas, renderer, **kwds):
        A = getattr(canvas, 'ctm', None)
        if not A:
            return
        x1 = node.x
        y1 = node.y
        x2 = x1 + node.width
        y2 = y1 + node.height
        D = kwds.copy()
        D['rect'] = DrawTimeCollector.transformAndFlatten(A, ((x1, y1), (x2, y2)))
        return D

    @staticmethod
    def transformAndFlatten(A, p):
        """ transform an flatten a list of points
        A   transformation matrix
        p   points [(x0,y0),....(xk,yk).....]
        """
        if tuple(A) != (1, 0, 0, 1, 0, 0):
            iA = inverse(A)
            p = transformPoints(iA, p)
        return tuple(flatten(p))

    @property
    def pmcanv(self):
        if not self._pmcanv:
            import renderPM
            self._pmcanv = renderPM.PMCanvas(1, 1)
        return self._pmcanv

    def wedgeDrawTimeCallback(self, node, canvas, renderer, **kwds):
        A = getattr(canvas, 'ctm', None)
        if not A:
            return
        if isinstance(node, Ellipse):
            c = self.pmcanv
            c.ellipse(node.cx, node.cy, node.rx, node.ry)
            p = c.vpath
            p = [(x[1], x[2]) for x in p]
        else:
            p = node.asPolygon().points
            p = [(p[i], p[i + 1]) for i in range(0, len(p), 2)]
        D = kwds.copy()
        D['poly'] = self.transformAndFlatten(A, p)
        return D

    def save(self, fnroot):
        """
        save the current information known to this collector
        fnroot is the root name of a resource to name the saved info
        override this to get the right semantics for your collector
        """
        import pprint
        f = open(fnroot + '.default-collector.out', 'w')
        try:
            pprint.pprint(self._info, f)
        finally:
            f.close()