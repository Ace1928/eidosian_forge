import time
from string import ascii_lowercase
from .gui import tkMessageBox
from .vertex import Vertex
from .arrow import Arrow, default_arrow_params
from .crossings import Crossing, ECrossing
from .smooth import TikZPicture
def DT_alpha(self):
    """
        Displays an alphabetical Dowker-Thistlethwaite code, as used in
        the knot tabulations.
        """
    code = self.DT_code(alpha=True)
    if code:
        self.write_text('DT: %s' % code)