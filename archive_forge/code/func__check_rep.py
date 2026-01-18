from ..sage_helper import _within_sage, sage_method
from .. import SnapPy
def _check_rep(self):
    assert all((self(g + g.swapcase()) == 1 for g in self.generators))
    assert all((self(rel) == 1 for rel in self.relators))