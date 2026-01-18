import copy
import re
import snappy_manifolds
from collections import OrderedDict, namedtuple
from .. import graphs
from .ordered_set import OrderedSet
def _component_starts_from_PD(self, code, labels, gluings):
    """
        A PD code determines an order and orientation on the link
        components as follows, where we view the code as labels on the
        strands at the point where two crossings are stuck together.

        1.  The minimum label on each component is used to order the
            components.

        2.  Each component is oriented by finding its minimal label,
            looking at the labels of its two neighbors, and then
            orienting the component towards the smaller of those two.

        This is designed so that a PLink-generated PD_code results in a
        link with the same component order and orientation.
        """
    starts = []
    while labels:
        m = min(labels)
        labels.remove(m)
        (c1, index1), (c2, index2) = gluings[m]
        if c1 == c2:
            next_label = min(set(code[c1]) - set([m]))
            direction = (c1, code[c1].index(next_label))
            starts.append(direction)
        else:
            j1, j2 = ((index1 + 2) % 4, (index2 + 2) % 4)
            l1, l2 = (code[c1][j1], code[c2][j2])
            if l1 < l2:
                next_label = l1
                direction = (c1, j1)
            elif l2 < l1:
                next_label = l2
                direction = (c2, j2)
            else:
                next_label = l1
                if code[c2][0] == l1 or code[c1][0] == m:
                    direction = (c1, j1)
                else:
                    direction = (c2, j2)
            starts.append(direction)
        while next_label != m:
            labels.remove(next_label)
            g = gluings[next_label]
            other_direction = g[1 - g.index(direction)]
            direction = (other_direction[0], (other_direction[1] + 2) % 4)
            next_label = code[direction[0]][direction[1]]
    return starts