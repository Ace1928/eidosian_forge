import copy
import re
import snappy_manifolds
from collections import OrderedDict, namedtuple
from .. import graphs
from .ordered_set import OrderedSet
class Crossing:
    """
    See "doc.pdf" for the conventions.  The sign of a crossing can be in {0,
    +1, -1}.  In the first case, the strands at the crossings can have
    any orientations, but crossings with sign +/-1 must be oriented as
    shown in "doc.pdf".

    Roles of some of the other attributes:

    * label: Arbitrary name used for printing the crossing.

    * directions: store the orientations of the link components passing
      through the crossing.  For a +1 crossing this is { (0, 2), (3, 1) }.
      Set with calls to make_tail.

    * adjacent: The other Crossings that this Crossing is attached to.

    * strand_labels: Numbering of the strands, used for DT codes and
      such.

    * strand_components: Which element of the parent
      Link.link_components each input's strand belongs to.
    """

    def __init__(self, label=None):
        self.label = label
        self.adjacent = CyclicList4()
        self._clear()
        self._adjacent_len = 4

    def _clear(self):
        self.sign, self.directions = (0, set())
        self._clear_strand_info()

    def _clear_strand_info(self):
        self.strand_labels = CyclicList4()
        self.strand_components = CyclicList4()

    def make_tail(self, a):
        """
        Orients the strand joining input "a" to input" a+2" to start at "a" and end at
        "a+2".
        """
        b = (a, (a + 2) % 4)
        if (b[1], b[0]) in self.directions:
            raise ValueError('Can only orient a strand once.')
        self.directions.add(b)

    def rotate(self, s):
        """
        Rotate the incoming connections by 90*s degrees anticlockwise.
        """

        def rotate(v):
            return (v + s) % 4
        new_adjacent = [self.adjacent[rotate(i)] for i in range(4)]
        for i, (o, j) in enumerate(new_adjacent):
            if o != self:
                o.adjacent[j] = (self, i)
                self.adjacent[i] = (o, j)
            else:
                self.adjacent[i] = (self, (j - s) % 4)
        self.directions = set(((rotate(a), rotate(b)) for a, b in self.directions))

    def rotate_by_90(self):
        """Effectively switches the crossing"""
        self.rotate(1)

    def rotate_by_180(self):
        """Effective reverses directions of the strands"""
        self.rotate(2)

    def orient(self):
        if (2, 0) in self.directions:
            self.rotate_by_180()
        self.sign = 1 if (3, 1) in self.directions else -1

    def is_incoming(self, i):
        if self.sign == 1:
            return i in (0, 3)
        elif self.sign == -1:
            return i in (0, 1)
        else:
            raise ValueError('Crossing not oriented')

    def __getitem__(self, i):
        return (self, i % 4)

    def entry_points(self):
        verts = [0, 1] if self.sign == -1 else [0, 3]
        return [CrossingEntryPoint(self, v) for v in verts]

    def crossing_strands(self):
        return [CrossingStrand(self, v) for v in range(4)]

    def __setitem__(self, i, other):
        o, j = other
        self.adjacent[i % 4] = other
        other[0].adjacent[other[1]] = (self, i % 4)

    def __repr__(self):
        return '%s' % (self.label,)

    def info(self):

        def format_adjacent(a):
            return (a[0].label, a[1]) if a else None
        print('<%s : %s : %s : %s>' % (self.label, self.sign, [format_adjacent(a) for a in self.adjacent], self.directions))

    def DT_info(self):
        """
        Returns (first label, second label, flip)
        """
        labels = self.strand_labels
        over = labels[3] + 1 if self.sign == 1 else labels[1] + 1
        under = labels[0] + 1
        if self.sign == 1:
            flip = 1 if labels[0] < labels[3] else 0
        else:
            flip = 0 if labels[0] < labels[1] else 1
        return (under, -over, flip) if over % 2 == 0 else (over, under, flip)

    def peer_info(self):
        labels = self.strand_labels
        SW = labels[0] if self.sign == 1 else labels[1]
        NW = labels[3] if self.sign == 1 else labels[0]
        if SW % 2 == 0:
            ans = (SW, (-NW, -self.sign))
        else:
            ans = (NW, (SW, self.sign))
        return ans