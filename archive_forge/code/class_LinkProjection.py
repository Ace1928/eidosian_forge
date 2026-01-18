import spherogram
import random
import itertools
from . import pl_utils
from .rational_linear_algebra import QQ, Matrix, Vector3
from .exceptions import GeneralPositionError
class LinkProjection:
    """
    The idea is to apply a unimodular matrix and to project the knot
    onto the z=0 plane. We then recover the crossing data by finding
    points that intersect after projecting then taking the over
    crossing to be the lift with larger z coord.

    >>> pts = fig8_points()
    >>> kp = LinkProjection(pts)
    >>> K = kp.link()
    >>> K.exterior().identify()
    [m004(0,0), 4_1(0,0), K2_1(0,0), K4a1(0,0), otet02_00001(0,0)]

    >>> M = Matrix([[0,1,1],[1,1,0],[0,0,2]])
    >>> kp = LinkProjection(pts, M)
    >>> K = kp.link()
    >>> K.exterior().identify()
    [m004(0,0), 4_1(0,0), K2_1(0,0), K4a1(0,0), otet02_00001(0,0)]

    >>> kp = LinkProjection(twist_knot_points())
    >>> K = kp.link()
    >>> M = Manifold('K5a1')
    >>> isos = M.is_isometric_to(K.exterior(), True)
    >>> {iso.cusp_maps()[0].det() for iso in isos}
    {1}
    """

    def __init__(self, points_by_component, mat=None):
        if mat is None:
            mat = Matrix([[0, -2, -1], [1, 2, 2], [-1, 1, 0]])
        self.mat = mat
        components = []
        for component_points in points_by_component:
            start = sum((len(C) for C in components))
            comp_len = len(component_points)
            components.append(list(range(start, start + comp_len)))
        self.components = components
        all_points = sum(points_by_component, [])
        self.points = [mat * p for p in all_points]
        self.projected_points = [proj(p) for p in self.points]
        assert min_dist_sq(self.points) > 1e-15
        if min_dist_sq(self.projected_points) < 1e-16:
            raise GeneralPositionError('Projection is nearly degenerate')
        self._setup_crossings()

    def _setup_crossings(self):
        pts = self.points
        crossings, arcs = ([], [])
        for component in self.components:
            successive_pairs = [(c, component[(i + 1) % len(component)]) for i, c in enumerate(component)]
            arcs += [Arc(pts[i], i, pts[j], j, i) for i, j in successive_pairs]
        for A, B in itertools.combinations(arcs, 2):
            a = (proj(A[0]), proj(A[1]))
            b = (proj(B[0]), proj(B[1]))
            if A.j == B.i:
                continue
            elif B.j == A.i:
                continue
            elif pl_utils.segments_meet_not_at_endpoint(a, b):
                M = Matrix([a[1] - a[0], b[0] - b[1]]).transpose()
                if M.rank() != 2:
                    raise GeneralPositionError('Segments overlap on their interiors')
                s, t = M.solve_right(b[0] - a[0])
                e = 1e-12
                if not (e < s < 1 - e and e < t < 1 - e):
                    raise GeneralPositionError('Intersection too near the end of one segment')
                x_a = (1 - s) * A[0] + s * A[1]
                x_b = (1 - t) * B[0] + t * B[1]
                assert norm_sq(proj(x_a - x_b)) < 1e-05
                height_a = x_a[2]
                height_b = x_b[2]
                assert abs(height_a - height_b) > 1e-14
                if height_a > height_b:
                    crossings.append(Crossing(A, B, s, t, len(crossings)))
                else:
                    crossings.append(Crossing(B, A, t, s, len(crossings)))
        self.crossings, self.arcs = (crossings, arcs)

    def link(self):
        strands = [spherogram.Strand(label='S%d' % i) for i, p in enumerate(self.points)]
        crossings = [spherogram.Crossing(label='C%d' % i) for i, c in enumerate(self.crossings)]
        for arc in self.arcs:
            A, a = (strands[arc.i], 1)
            for t, C in arc.crossings:
                B = crossings[C.label]
                if arc == C.over:
                    b = 3 if C.sign == 1 else 1
                else:
                    b = 0
                A[a] = B[b]
                A, a = (B, (b + 2) % 4)
            B, b = (strands[arc.j], 0)
            A[a] = B[b]
        L = spherogram.Link(strands + crossings)
        return L