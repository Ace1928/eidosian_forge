from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.location import FeatureLibLocation
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import byteord, tobytes
from collections import OrderedDict
import itertools
class BaseAxis(Statement):
    """An axis definition, being either a ``VertAxis.BaseTagList/BaseScriptList``
    pair or a ``HorizAxis.BaseTagList/BaseScriptList`` pair."""

    def __init__(self, bases, scripts, vertical, location=None):
        Statement.__init__(self, location)
        self.bases = bases
        self.scripts = scripts
        self.vertical = vertical

    def build(self, builder):
        """Calls the builder object's ``set_base_axis`` callback."""
        builder.set_base_axis(self.bases, self.scripts, self.vertical)

    def asFea(self, indent=''):
        direction = 'Vert' if self.vertical else 'Horiz'
        scripts = ['{} {} {}'.format(a[0], a[1], ' '.join(map(str, a[2]))) for a in self.scripts]
        return '{}Axis.BaseTagList {};\n{}{}Axis.BaseScriptList {};'.format(direction, ' '.join(self.bases), indent, direction, ', '.join(scripts))