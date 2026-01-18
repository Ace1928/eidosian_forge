import time
from string import ascii_lowercase
from .gui import tkMessageBox
from .vertex import Vertex
from .arrow import Arrow, default_arrow_params
from .crossings import Crossing, ECrossing
from .smooth import TikZPicture
def BB_framing(self):
    """
        Return the standard meridian-longitude coordinates of the
        blackboard longitude (i.e. the peripheral element obtained
        by following the top of a tubular neighborhood of the knot).
        """
    try:
        components = self.crossing_components()
    except ValueError:
        return None
    framing = []
    for component in components:
        m = 0
        for ec in component:
            crossing = ec.crossing
            if crossing.comp1 == crossing.comp2 == component:
                if ec.crossing.sign() == 'RH':
                    m += 1
                elif ec.crossing.sign() == 'LH':
                    m -= 1
        framing.append((m // 2, 1))
    return framing