import time
from itertools import chain
from xml.dom import minidom
import xml.etree.ElementTree as ET
class Graphics:
    """An Entry subelement used to represents the visual representation.

    A subelement of Entry, specifying its visual representation, as
    described in release KGML v0.7.2 (http://www.kegg.jp/kegg/xml/docs/)

    Attributes:
     - name         Label for the graphics object
     - x            X-axis position of the object (int)
     - y            Y-axis position of the object (int)
     - coords       polyline coordinates, list of (int, int) tuples
     - type         object shape
     - width        object width (int)
     - height       object height (int)
     - fgcolor      object foreground color (hex RGB)
     - bgcolor      object background color (hex RGB)

    Some attributes are present only for specific graphics types.  For
    example, line types do not (typically) have a width.
    We permit non-DTD attributes and attribute settings, such as

    dash         List of ints, describing an on/off pattern for dashes

    """

    def __init__(self, parent):
        """Initialize the class."""
        self.name = ''
        self._x = None
        self._y = None
        self._coords = None
        self.type = ''
        self._width = None
        self._height = None
        self.fgcolor = ''
        self.bgcolor = ''
        self._parent = parent

    def _getx(self):
        return self._x

    def _setx(self, value):
        self._x = float(value)

    def _delx(self):
        del self._x
    x = property(_getx, _setx, _delx, 'The X coordinate for the graphics element.')

    def _gety(self):
        return self._y

    def _sety(self, value):
        self._y = float(value)

    def _dely(self):
        del self._y
    y = property(_gety, _sety, _dely, 'The Y coordinate for the graphics element.')

    def _getwidth(self):
        return self._width

    def _setwidth(self, value):
        self._width = float(value)

    def _delwidth(self):
        del self._width
    width = property(_getwidth, _setwidth, _delwidth, 'The width of the graphics element.')

    def _getheight(self):
        return self._height

    def _setheight(self, value):
        self._height = float(value)

    def _delheight(self):
        del self._height
    height = property(_getheight, _setheight, _delheight, 'The height of the graphics element.')

    def _getcoords(self):
        return self._coords

    def _setcoords(self, value):
        clist = [int(e) for e in value.split(',')]
        self._coords = [tuple(clist[i:i + 2]) for i in range(0, len(clist), 2)]

    def _delcoords(self):
        del self._coords
    coords = property(_getcoords, _setcoords, _delcoords, 'Polyline coordinates for the graphics element.')

    def _getfgcolor(self):
        return self._fgcolor

    def _setfgcolor(self, value):
        if value == 'none':
            self._fgcolor = '#000000'
        else:
            self._fgcolor = value

    def _delfgcolor(self):
        del self._fgcolor
    fgcolor = property(_getfgcolor, _setfgcolor, _delfgcolor, 'Foreground color.')

    def _getbgcolor(self):
        return self._bgcolor

    def _setbgcolor(self, value):
        if value == 'none':
            self._bgcolor = '#000000'
        else:
            self._bgcolor = value

    def _delbgcolor(self):
        del self._bgcolor
    bgcolor = property(_getbgcolor, _setbgcolor, _delbgcolor, 'Background color.')

    @property
    def element(self):
        """Return the Graphics as a valid KGML element."""
        graphics = ET.Element('graphics')
        if isinstance(self.fgcolor, str):
            fghex = self.fgcolor
        else:
            fghex = '#' + self.fgcolor.hexval()[2:]
        if isinstance(self.bgcolor, str):
            bghex = self.bgcolor
        else:
            bghex = '#' + self.bgcolor.hexval()[2:]
        graphics.attrib = {'name': self.name, 'type': self.type, 'fgcolor': fghex, 'bgcolor': bghex}
        for n, attr in [('x', '_x'), ('y', '_y'), ('width', '_width'), ('height', '_height')]:
            if getattr(self, attr) is not None:
                graphics.attrib[n] = str(getattr(self, attr))
        if self.type == 'line':
            graphics.attrib['coords'] = ','.join([str(e) for e in chain.from_iterable(self.coords)])
        return graphics

    @property
    def bounds(self):
        """Coordinate bounds for the Graphics element.

        Return the bounds of the Graphics object as an [(xmin, ymin),
        (xmax, ymax)] tuple.  Coordinates give the centre of the
        circle, rectangle, roundrectangle elements, so we have to
        adjust for the relevant width/height.
        """
        if self.type == 'line':
            xlist = [x for x, y in self.coords]
            ylist = [y for x, y in self.coords]
            return [(min(xlist), min(ylist)), (max(xlist), max(ylist))]
        else:
            return [(self.x - self.width * 0.5, self.y - self.height * 0.5), (self.x + self.width * 0.5, self.y + self.height * 0.5)]

    @property
    def centre(self):
        """Return the centre of the Graphics object as an (x, y) tuple."""
        return (0.5 * (self.bounds[0][0] + self.bounds[1][0]), 0.5 * (self.bounds[0][1] + self.bounds[1][1]))