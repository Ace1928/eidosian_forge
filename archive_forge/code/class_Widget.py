from reportlab.graphics import shapes
from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from weakref import ref as weakref_ref
class Widget(PropHolder, shapes.UserNode):
    """Base for all user-defined widgets.  Keep as simple as possible. Does
    not inherit from Shape so that we can rewrite shapes without breaking
    widgets and vice versa."""

    def _setKeywords(self, **kw):
        for k, v in kw.items():
            if k not in self.__dict__:
                setattr(self, k, v)

    def draw(self):
        msg = 'draw() must be implemented for each Widget!'
        raise NotImplementedError(msg)

    def demo(self):
        msg = 'demo() must be implemented for each Widget!'
        raise NotImplementedError(msg)

    def provideNode(self):
        return self.draw()

    def getBounds(self):
        """Return outer boundary as x1,y1,x2,y2.  Can be overridden for efficiency"""
        return self.draw().getBounds()