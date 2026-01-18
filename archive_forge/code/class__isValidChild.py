import codecs, re
from reportlab.lib.utils import isSeq, isBytes, isStr
from reportlab.lib import colors
from reportlab.lib.normalDate import NormalDate
class _isValidChild(Validator):
    """ValidChild validator class."""

    def test(self, x):
        """Is this child allowed in a drawing or group?
        I.e. does it descend from Shape or UserNode?
        """
        from reportlab.graphics.shapes import UserNode, Shape
        return isinstance(x, UserNode) or isinstance(x, Shape)