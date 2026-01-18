from .component import ZeroDimensionalComponent
from .rur import RUR
from . import matrix
from . import findLoops
from . import utilities
from ..sage_helper import _within_sage
from ..pari import Gen, pari
import re
class Flattenings(dict):
    """
    Represents a flattening assigned to each edge of a simplex as dictionary.

    We assign to each pair of parallel edges of each simplex a triple (w, z, p)
    such that::

           w = log(z) + p * (2 * pi * i / N)   where N is fixed and even.

    For N = 2, the three triples belonging to a simplex form a combinatorial
    flattening (w0, w1, w2) as defined in Definition 3.1 in
    Walter D. Neumann, Extended Bloch group and the Cheeger-Chern-Simons class
    http://arxiv.org/abs/math.GT/0307092

    For N > 2, the three triples form a generalized combinatorial flattening
    (w0, w1, w2) that gives an element in the generalized Extended Bloch group
    which is the Extended Bloch group corresponding to the Riemann surface
    given by::

                 u1 * e^w0 + u2 * e^w1 = 1

    where u1^N = u2^N = 1.

    A representation in SL(n,C) and SL(n,C)/{+1,-1} with n even gives an element
    in the generalized Extended Bloch group for N = 2.
    A representation in PSL(n,C) with n even in the group for N = n.
    A representation in PSL(n,C) with n odd in the group for N = 2 * n.

    This work has not been published yet.

    If f is a flattening, then in the notation of Neumann, the value of::

        f['z_xxxx_y']    is (w0, z, p)
        f['zp_xxxx_y']   is (w1, z', q)
        f['zpp_xxxx_y']  is (w2, z'', r).
    """

    def __init__(self, d, manifold_thunk=lambda: None, evenN=2):
        super(Flattenings, self).__init__(d)
        self._is_numerical = True
        self._manifold_thunk = manifold_thunk
        self.dimension = 0
        self._evenN = evenN

    def __repr__(self):
        dict_repr = dict.__repr__(self)
        return 'Flattenings(%s, ...)' % dict_repr

    def _repr_pretty_(self, p, cycle):
        if cycle:
            p.text('Flattenings(...)')
        else:
            with p.group(4, 'Flattenings(', ')'):
                p.breakable()
                p.pretty(dict(self))
                p.text(', ...')

    def get_manifold(self):
        """
        Get the manifold for which this structure represents a flattening.
        """
        return self._manifold_thunk()

    def num_tetrahedra(self):
        """
        The number of tetrahedra for which we have cross ratios.
        """
        return _num_tetrahedra(self)

    def N(self):
        """
        Get the N such that these cross ratios are for
        SL/PSL(N,C)-representations.
        """
        return _N_for_shapes(self)

    @classmethod
    def from_tetrahedra_shapes_of_manifold(cls, M):
        """
        Takes as argument a manifold and produces (weak) flattenings using
        the tetrahedra_shapes of the manifold M.

        >>> from snappy import Manifold
        >>> M = Manifold("5_2")
        >>> flattenings = Flattenings.from_tetrahedra_shapes_of_manifold(M)
        >>> flattenings.check_against_manifold(M)
        >>> flattenings.check_against_manifold()
        """
        PiI = pari('Pi * I')
        num_tets = M.num_tetrahedra()
        z_cross_ratios = M.tetrahedra_shapes(part='rect', dec_prec=pari.get_real_precision())
        all_cross_ratios = sum([[z, 1 / (1 - z), 1 - 1 / z] for z in z_cross_ratios], [])
        log_all_cross_ratios = [z.log() for z in all_cross_ratios]

        def flattening_condition(r):
            return 3 * r * [0] + 3 * [1] + 3 * (num_tets - r - 1) * [0]
        flattening_conditions = [flattening_condition(r) for r in range(num_tets)]
        try:
            equations = M.gluing_equations().data
        except AttributeError:
            equations = [[int(c) for c in row] for row in M.gluing_equations().rows()]
        all_equations = equations + flattening_conditions
        u, v, d_mat = matrix.smith_normal_form(all_equations)
        extra_cols = len(all_equations[0]) - len(all_equations)
        d = [d_mat[r][r + extra_cols] for r in range(len(d_mat))]
        errors = matrix.matrix_mult_vector(all_equations, log_all_cross_ratios)
        int_errors = [(x / PiI).real().round() for x in errors]
        int_errors_in_other_basis = matrix.matrix_mult_vector(u, int_errors)

        def quotient(x, y):
            if x == 0 and y == 0:
                return 0
            assert x % y == 0, '%s %s' % (x, y)
            return x / y
        flattenings_in_other_basis = extra_cols * [0] + [-quotient(x, y) for x, y in zip(int_errors_in_other_basis, d)]
        flattenings = matrix.matrix_mult_vector(v, flattenings_in_other_basis)
        assert matrix.matrix_mult_vector(all_equations, flattenings) == [-x for x in int_errors]
        keys = sum([['z_0000_%d' % i, 'zp_0000_%d' % i, 'zpp_0000_%d' % i] for i in range(num_tets)], [])
        Mcopy = M.copy()
        return Flattenings(dict([(k, (log + PiI * p, z, p)) for k, log, z, p in zip(keys, log_all_cross_ratios, all_cross_ratios, flattenings)]), manifold_thunk=lambda: Mcopy)

    def get_order(self):
        """
        Returns the number N. This flattening represents an element in the
        generalized Extended Bloch group for the Riemann surface given by
        u1 * e^w0 + u2 * e^w1 = 1 where u1^N = u2^N = 1.
        """
        return self._evenN

    def get_zpq_triple(self, key_z):
        """
        Gives a flattening as triple [z;p,q] representing an element
        in the generalized Extended Bloch group similar to the way the
        triple [z;p,q] is used in Lemma 3.2 in
        Walter D. Neumann, Extended Bloch group and the Cheeger-Chern-Simons class
        http://arxiv.org/abs/math.GT/0307092
        """
        if not key_z[:2] == 'z_':
            raise Exception('Need to be called with cross ratio variable z_....')
        key_zp = 'zp_' + key_z[2:]
        w, z, p = self[key_z]
        wp, zp, q_canonical_branch_cut = self[key_zp]
        pari_z = _convert_to_pari_float(z)
        f = pari('2 * Pi * I') / self._evenN
        q_dilog_branch_cut = ((wp + (1 - pari_z).log()) / f).round()
        return (z, p, q_dilog_branch_cut)

    def complex_volume(self, with_modulo=False):
        """
        Compute complex volume. The complex volume is defined only up to
        some multiple of m where m = i * pi**2/6 for PSL(2,C) and SL(N,C)
        and m = i * pi**2/18 for PSL(3,C).

        When called with with_modulo = True, gives a pair
        (volume, m)
        """
        if self._evenN == 2:
            m = pari('Pi^2/6')
        else:
            m = pari('Pi^2/18')
        sum_L_functions = sum([_L_function(self.get_zpq_triple(key), self._evenN) for key in list(self.keys()) if key[:2] == 'z_'])
        cvol = sum_L_functions / pari('I')
        vol = cvol.real()
        cs = cvol.imag() % m
        if cs > m / 2 + pari('1e-12'):
            cs = cs - m
        cvol = vol + cs * pari('I')
        if with_modulo:
            if self._evenN not in [2, 6]:
                raise Exception('Unknown torsion')
            return (cvol, m * pari('I'))
        return cvol

    def check_against_manifold(self, M=None, epsilon=1e-10):
        """
        Checks that the flattening really is a solution to the logarithmic
        PGL(N,C) gluing equations of a manifold. Usage similar to
        check_against_manifold of Ptolemy Coordinates, see
        help(ptolemy.Coordinates) for similar examples.

        === Arguments ===

        M --- manifold to check this for
        epsilon --- maximal allowed error when checking the equations
        """
        if M is None:
            M = self.get_manifold()
        if M is None:
            raise Exception('Need to give manifold')
        f = pari('2 * Pi * I') / self._evenN
        for w, z, p in list(self.values()):
            _check_relation(w - (z.log() + f * p), epsilon, 'Flattening relation w == log(z) + PiI * p')
        for k in list(self.keys()):
            if k[:2] == 'z_':
                w, z, p = self[k]
                wp, zp, q = self['zp_' + k[2:]]
                wpp, zpp, r = self['zpp_' + k[2:]]
                _check_relation(w + wp + wpp, epsilon, 'Flattening relation w0 + w1 + w2 == 0')
        some_z = list(self.keys())[0]
        variable_name, index, tet_index = some_z.split('_')
        if variable_name not in ['z', 'zp', 'zpp']:
            raise Exception('Variable not z, zp, or, zpp')
        if len(index) != 4:
            raise Exception('Not 4 indices')
        N = sum([int(x) for x in index]) + 2
        matrix_with_explanations = M.gluing_equations_pgl(N, equation_type='all')
        matrix = matrix_with_explanations.matrix
        rows = matrix_with_explanations.explain_rows
        cols = matrix_with_explanations.explain_columns
        for row in range(len(rows)):
            s = 0
            for col in range(len(cols)):
                flattening_variable = cols[col]
                w, z, p = self[flattening_variable]
                s = s + w
            _check_relation(s, epsilon, 'Gluing equation %s' % rows[row])