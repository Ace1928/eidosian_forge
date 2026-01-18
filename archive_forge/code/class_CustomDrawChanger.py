from time import mktime, gmtime, strftime
from math import log10, pi, floor, sin, cos, hypot
import weakref
from reportlab.graphics.shapes import transformPoints, inverse, Ellipse, Group, String, numericXShift
from reportlab.lib.utils import flatten
from reportlab.pdfbase.pdfmetrics import stringWidth
class CustomDrawChanger:
    """
    a class to simplify making changes at draw time
    """

    def __init__(self):
        self.store = None

    def __call__(self, change, obj):
        if change:
            self.store = self._changer(obj)
            assert isinstance(self.store, dict), '%s.changer should return a dict of changed attributes' % self.__class__.__name__
        elif self.store is not None:
            for a, v in self.store.items():
                setattr(obj, a, v)
            self.store = None

    def _changer(self, obj):
        """
        When implemented this method should return a dictionary of
        original attribute values so that a future self(False,obj)
        can restore them.
        """
        raise RuntimeError('Abstract method _changer called')