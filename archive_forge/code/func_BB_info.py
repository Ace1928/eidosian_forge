import time
from string import ascii_lowercase
from .gui import tkMessageBox
from .vertex import Vertex
from .arrow import Arrow, default_arrow_params
from .crossings import Crossing, ECrossing
from .smooth import TikZPicture
def BB_info(self):
    """
        Displays the meridian-longitude coordinates of the blackboard
        longitudes of the components of the link
        """
    framing = self.BB_framing()
    if framing:
        self.write_text(('BB framing:  %s' % framing).replace(', ', ','))