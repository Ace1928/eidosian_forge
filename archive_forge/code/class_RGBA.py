import sys
import warnings
from ..overrides import override, strip_boolean_result
from ..module import get_introspection_module
from gi import PyGIDeprecationWarning, require_version
class RGBA(Gdk.RGBA):

    def __init__(self, red=1.0, green=1.0, blue=1.0, alpha=1.0):
        Gdk.RGBA.__init__(self)
        self.red = red
        self.green = green
        self.blue = blue
        self.alpha = alpha

    def __eq__(self, other):
        if not isinstance(other, Gdk.RGBA):
            return False
        return self.equal(other)

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return 'Gdk.RGBA(red=%f, green=%f, blue=%f, alpha=%f)' % (self.red, self.green, self.blue, self.alpha)

    def __iter__(self):
        """Iterator which allows easy conversion to tuple and list types."""
        yield self.red
        yield self.green
        yield self.blue
        yield self.alpha

    def to_color(self):
        """Converts this RGBA into a Color instance which excludes alpha."""
        return Color(int(self.red * Color.MAX_VALUE), int(self.green * Color.MAX_VALUE), int(self.blue * Color.MAX_VALUE))

    @classmethod
    def from_color(cls, color):
        """Returns a new RGBA instance given a Color instance."""
        return cls(color.red_float, color.green_float, color.blue_float)