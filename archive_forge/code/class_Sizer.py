from reportlab.graphics import shapes
from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from weakref import ref as weakref_ref
class Sizer(Widget):
    """Container to show size of all enclosed objects"""
    _attrMap = AttrMap(BASE=shapes.SolidShape, contents=AttrMapValue(isListOfShapes, desc='Contained drawable elements'))

    def __init__(self, *elements):
        self.contents = []
        self.fillColor = colors.cyan
        self.strokeColor = colors.magenta
        for elem in elements:
            self.add(elem)

    def _addNamedNode(self, name, node):
        """if name is not None add an attribute pointing to node and add to the attrMap"""
        if name:
            if name not in list(self._attrMap.keys()):
                self._attrMap[name] = AttrMapValue(isValidChild)
            setattr(self, name, node)

    def add(self, node, name=None):
        """Appends non-None child node to the 'contents' attribute. In addition,
        if a name is provided, it is subsequently accessible by name
        """
        if node is not None:
            assert isValidChild(node), 'Can only add Shape or UserNode objects to a Group'
            self.contents.append(node)
            self._addNamedNode(name, node)

    def getBounds(self):
        if self.contents:
            b = []
            for elem in self.contents:
                b.append(elem.getBounds())
            return shapes.getRectsBounds(b)
        else:
            return (0, 0, 0, 0)

    def draw(self):
        g = shapes.Group()
        x1, y1, x2, y2 = self.getBounds()
        r = shapes.Rect(x=x1, y=y1, width=x2 - x1, height=y2 - y1, fillColor=self.fillColor, strokeColor=self.strokeColor)
        g.add(r)
        for elem in self.contents:
            g.add(elem)
        return g