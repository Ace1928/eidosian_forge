import time
from string import ascii_lowercase
from .gui import tkMessageBox
from .vertex import Vertex
from .arrow import Arrow, default_arrow_params
from .crossings import Crossing, ECrossing
from .smooth import TikZPicture
def DT_normal(self):
    """
        Displays a Dowker-Thistlethwaite code as a list of tuples of
        signed even integers.
        """
    code = self.DT_code()
    if code:
        self.write_text(('DT: %s,  %s' % code).replace(', ', ','))