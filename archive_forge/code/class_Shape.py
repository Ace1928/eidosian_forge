import os, sys
from math import pi, cos, sin, sqrt, radians, floor
from reportlab.platypus import Flowable
from reportlab.rl_config import shapeChecking, verbose, defaultGraphicsFontName as _baseGFontName, _unset_, decimalSymbol
from reportlab.lib import logger
from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.utils import isSeq, asBytes
from reportlab.lib.attrmap import *
from reportlab.lib.rl_accel import fp_str
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.fonts import tt2ps
from reportlab.pdfgen.canvas import FILL_EVEN_ODD, FILL_NON_ZERO
from . transform import *
class Shape(_SetKeyWordArgs, _DrawTimeResizeable):
    """Base class for all nodes in the tree. Nodes are simply
    packets of data to be created, stored, and ultimately
    rendered - they don't do anything active.  They provide
    convenience methods for verification but do not
    check attribiute assignments or use any clever setattr
    tricks this time."""
    _attrMap = AttrMap()

    def copy(self):
        """Return a clone of this shape."""
        raise NotImplementedError('No copy method implemented for %s' % self.__class__.__name__)

    def getProperties(self, recur=1):
        """Interface to make it easy to extract automatic
        documentation"""
        props = {}
        for key, value in self.__dict__.items():
            if key[0:1] != '_':
                props[key] = value
        return props

    def setProperties(self, props):
        """Supports the bulk setting if properties from,
        for example, a GUI application or a config file."""
        self.__dict__.update(props)

    def dumpProperties(self, prefix=''):
        """Convenience. Lists them on standard output.  You
        may provide a prefix - mostly helps to generate code
        samples for documentation."""
        propList = list(self.getProperties().items())
        propList.sort()
        if prefix:
            prefix = prefix + '.'
        for name, value in propList:
            print('%s%s = %s' % (prefix, name, value))

    def verify(self):
        """If the programmer has provided the optional
        _attrMap attribute, this checks all expected
        attributes are present; no unwanted attributes
        are present; and (if a checking function is found)
        checks each attribute.  Either succeeds or raises
        an informative exception."""
        if self._attrMap is not None:
            for key in self.__dict__.keys():
                if key[0] != '_':
                    assert key in self._attrMap, 'Unexpected attribute %s found in %s' % (key, self)
            for attr, metavalue in self._attrMap.items():
                assert hasattr(self, attr), 'Missing attribute %s from %s' % (attr, self)
                value = getattr(self, attr)
                assert metavalue.validate(value), 'Invalid value %s for attribute %s in class %s' % (value, attr, self.__class__.__name__)
    if shapeChecking:
        'This adds the ability to check every attribute assignment as it is made.\n        It slows down shapes but is a big help when developing. It does not\n        get defined if rl_config.shapeChecking = 0'

        def __setattr__(self, attr, value):
            """By default we verify.  This could be off
            in some parallel base classes."""
            validateSetattr(self, attr, value)

    def getBounds(self):
        """Returns bounding rectangle of object as (x1,y1,x2,y2)"""
        raise NotImplementedError('Shapes and widgets must implement getBounds')