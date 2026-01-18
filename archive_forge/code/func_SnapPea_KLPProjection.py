import time
from string import ascii_lowercase
from .gui import tkMessageBox
from .vertex import Vertex
from .arrow import Arrow, default_arrow_params
from .crossings import Crossing, ECrossing
from .smooth import TikZPicture
def SnapPea_KLPProjection(self):
    """
        Constructs a python simulation of a SnapPea KLPProjection
        (Kernel Link Projection) structure.  See Jeff Weeks' SnapPea
        file link_projection.h for definitions.  Here the KLPCrossings
        are modeled by dictionaries.  This method requires that all
        components be closed.  A side effect is that the KLP attributes
        of all crossings are updated.

        The following excerpt from link_projection.h describes the
        main convention:

        If you view a crossing (from above) so that the strands go in the
        direction of the positive x- and y-axes, then the strand going in
        the x-direction is the KLPStrandX, and the strand going in the
        y-direction is the KLPStrandY.  Note that this definition does not
        depend on which is the overstrand and which is the understrand:

        ::

                             KLPStrandY
                                 ^
                                 |
                             ----+---> KLPStrandX
                                 |
                                 |

                """
    try:
        components = self.crossing_components()
    except ValueError:
        return None
    num_crossings = len(self.Crossings)
    num_free_loops = 0
    num_components = len(components)
    id = lambda x: self.Crossings.index(x.crossing)
    for component in components:
        this_component = components.index(component)
        N = len(component)
        for n in range(N):
            this = component[n]
            previous = component[n - 1]
            next = component[(n + 1) % N]
            this.crossing.KLP['sign'] = sign = this.crossing.sign()
            if this.strand == 'X':
                this.crossing.KLP['Xbackward_neighbor'] = id(previous)
                this.crossing.KLP['Xbackward_strand'] = previous.strand
                this.crossing.KLP['Xforward_neighbor'] = id(next)
                this.crossing.KLP['Xforward_strand'] = next.strand
                this.crossing.KLP['Xcomponent'] = this_component
            else:
                this.crossing.KLP['Ybackward_neighbor'] = id(previous)
                this.crossing.KLP['Ybackward_strand'] = previous.strand
                this.crossing.KLP['Yforward_neighbor'] = id(next)
                this.crossing.KLP['Yforward_strand'] = next.strand
                this.crossing.KLP['Ycomponent'] = this_component
        if N == 0:
            num_free_loops += 1
    KLP_crossings = [crossing.KLP for crossing in self.Crossings]
    return (num_crossings, num_free_loops, num_components, KLP_crossings)