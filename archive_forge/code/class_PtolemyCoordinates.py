from .component import ZeroDimensionalComponent
from .rur import RUR
from . import matrix
from . import findLoops
from . import utilities
from ..sage_helper import _within_sage
from ..pari import Gen, pari
import re
class PtolemyCoordinates(dict):
    """
    Represents a solution of a Ptolemy variety as python dictionary.

    === Examples ===

    Construct solution from magma output:

    >>> from snappy.ptolemy.processMagmaFile import _magma_output_for_4_1__sl3, solutions_from_magma
    >>> from snappy import Manifold
    >>> solutions = solutions_from_magma(_magma_output_for_4_1__sl3)
    >>> solution = solutions[2]

    Access a Ptolemy coordinate:

    >>> solution['c_2100_0']
    1

    >>> solution.number_field()
    x^2 + x + 1

    Solution is always 0 dimensional:

    >>> solution.dimension
    0

    Check that it is really a solution, exactly:

    >>> solution.check_against_manifold()

    If the solution was not created through the ptolemy module
    and thus not associated to a manifold, we need to explicitly
    specify one:

    >>> myDict = {}
    >>> for key, value in solution.items():
    ...     myDict[key] = value
    >>> mysolution = PtolemyCoordinates(myDict)
    >>> M = Manifold("4_1")
    >>> solution.check_against_manifold(M)

    Turn into (Galois conjugate) numerical solutions:

    >>> old_precision = pari.set_real_precision(100) # with high precision
    >>> numerical_solutions = solution.numerical()

    Check that it is a solution, numerically:

    >>> numerical_solutions[0].check_against_manifold(M, 1e-80)
    >>> pari.set_real_precision(old_precision) # reset pari engine
    100

    Compute cross ratios from the Ptolemy coordinates (cross ratios
    according to SnapPy convention, see help(solution.cross_ratios):

    >>> cross = solution.cross_ratios()
    >>> cross['z_0001_0']
    Mod(-x, x^2 + x + 1)

    Compute volumes:

    >>> volumes = cross.volume_numerical()

    Check that volume is 4 times the geometric one:

    >>> volume = volumes[0].abs()
    >>> diff = abs(4 * M.volume() - volume)
    >>> diff < 1e-9
    True

    Compute flattenings:

    >>> flattenings = solution.flattenings_numerical()

    Compute complex volumes:

    >>> cvols = [flattening.complex_volume() for flattening in flattenings]
    >>> volume = cvols[0].real().abs()
    >>> chernSimons = cvols[0].imag()
    >>> diff = abs(4 * M.volume() - volume)
    >>> diff < 1e-9
    True

    >>> from snappy import pari
    >>> normalized = chernSimons * 6 / (pari('Pi')**2)

    Check that Chern Simons is zero up to 6 torsion:

    >>> normalized - normalized.round() < 1e-9
    True
    """

    def __init__(self, d, is_numerical=True, py_eval_section=None, manifold_thunk=lambda: None, non_trivial_generalized_obstruction_class=False):
        self._manifold_thunk = manifold_thunk
        self._is_numerical = is_numerical
        self.dimension = 0
        self._non_trivial_generalized_obstruction_class = non_trivial_generalized_obstruction_class
        processed_dict = d
        if py_eval_section is not None:
            processed_dict = py_eval_section['variable_dict'](d)
            if py_eval_section.get('non_trivial_generalized_obstruction_class'):
                self._non_trivial_generalized_obstruction_class = True
        self._edge_cache = {}
        self._matrix_cache = []
        self._inverse_matrix_cache = []
        super(PtolemyCoordinates, self).__init__(processed_dict)

    def __repr__(self):
        dict_repr = dict.__repr__(self)
        return 'PtolemyCoordinates(%s, is_numerical = %r, ...)' % (dict_repr, self._is_numerical)

    def _repr_pretty_(self, p, cycle):
        if cycle:
            p.text('PtolemyCoordinates(...)')
        else:
            with p.group(4, 'PtolemyCoordinates(', ')'):
                p.breakable()
                p.pretty(dict(self))
                p.text(',')
                p.breakable()
                p.text('is_numerical = %r, ...' % self._is_numerical)

    def get_manifold(self):
        """
        Get the manifold for which this structure represents a solution
        to the Ptolemy variety.
        """
        return self._manifold_thunk()

    def num_tetrahedra(self):
        """
        The number of tetrahedra for which we have Ptolemy coordinates.
        """
        return _num_tetrahedra(self)

    def N(self):
        """
        Get the *N* where these coordinates are for SL/PSL(*N*, **C**)-representations.
        """
        N, has_obstruction = _N_and_has_obstruction_for_ptolemys(self)
        return N

    def has_obstruction(self):
        """
        Whether the Ptolemy variety has legacy obstruction class that
        modifies the Ptolemy relation to
        """
        N, has_obstruction = _N_and_has_obstruction_for_ptolemys(self)
        return has_obstruction

    def number_field(self):
        """
        For an exact solution, return the number_field spanned by the
        Ptolemy coordinates. If number_field is Q, return None.
        """
        if self._is_numerical:
            raise ExactMethodError('number_field')
        return _get_number_field(self)

    def numerical(self):
        """
        Turn exact solutions into numerical solutions using pari.

        Take an exact solution:

        >>> from snappy.ptolemy.processMagmaFile import _magma_output_for_4_1__sl3, solutions_from_magma
        >>> solutions = solutions_from_magma(_magma_output_for_4_1__sl3)
        >>> solution = solutions[2]

        Turn into a numerical solution:

        >>> old_precision = pari.set_real_precision(100) # with high precision
        >>> numerical_solutions = solution.numerical()
        >>> pari.set_real_precision(old_precision) # reset pari engine
        100

        Pick one of the Galois conjugates:

        >>> numerical_solution = numerical_solutions[0]
        >>> value = numerical_solution['c_1110_0']
        """
        if self._is_numerical:
            return self
        return ZeroDimensionalComponent([PtolemyCoordinates(d, is_numerical=True, manifold_thunk=self._manifold_thunk, non_trivial_generalized_obstruction_class=self._non_trivial_generalized_obstruction_class) for d in _to_numerical(self)])

    def to_PUR(self):
        """
        If any Ptolemy coordinates are given as Rational Univariate
        Representation, convert them to Polynomial Univariate Representation and
        return the result.

        See to_PUR of RUR.

        This conversion might lead to very large coefficients.
        """
        return PtolemyCoordinates(_apply_to_RURs(self, RUR.to_PUR), is_numerical=self._is_numerical, manifold_thunk=self._manifold_thunk, non_trivial_generalized_obstruction_class=self._non_trivial_generalized_obstruction_class)

    def multiply_terms_in_RUR(self):
        """
        If a Ptolemy coordinate is given as Rational Univariate Representation
        with numerator and denominator being a product, multiply the terms and
        return the result.

        See multiply_terms of RUR.

        This loses information about how the numerator and denominator are
        factorised.
        """
        return PtolemyCoordinates(_apply_to_RURs(self, RUR.multiply_terms), is_numerical=self._is_numerical, manifold_thunk=self._manifold_thunk, non_trivial_generalized_obstruction_class=self._non_trivial_generalized_obstruction_class)

    def multiply_and_simplify_terms_in_RUR(self):
        """
        If a Ptolemy coordinate is given as Rational Univariate Representation
        with numerator and denominator being a product, multiply the terms,
        reduce the fraction and return the result.

        See multiply_and_simplify_terms of RUR.

        This loses information about how the numerator and denominator are
        factorised.

        """
        return PtolemyCoordinates(_apply_to_RURs(self, RUR.multiply_and_simplify_terms), is_numerical=self._is_numerical, manifold_thunk=self._manifold_thunk, non_trivial_generalized_obstruction_class=self._non_trivial_generalized_obstruction_class)

    def cross_ratios(self):
        """
        Compute cross ratios from Ptolemy coordinates. The cross ratios are
        according to the SnapPy convention, so we have::

             z = 1 - 1/zp, zp = 1 - 1/zpp, zpp = 1 - 1/z

        where::

             z   is at the edge 01 and equal to   s0 * s1 * (c_1010 * c_0101) / (c_1001 * c_0110)
             zp  is at the edge 02 and equal to - s0 * s2 * (c_1001 * c_0110) / (c_1100 * c_0011)
             zpp is at the edge 03 and equal to   s0 * s3 * (c_1100 * c_0011) / (c_0101 * c_1010).

        Note that this is different from the convention used in
        Garoufalidis, Goerner, Zickert:
        Gluing Equations for PGL(n,C)-Representations of 3-Manifolds
        http://arxiv.org/abs/1207.6711

        Take an exact solution:

        >>> from snappy.ptolemy.processMagmaFile import _magma_output_for_4_1__sl3, solutions_from_magma
        >>> solutions = solutions_from_magma(_magma_output_for_4_1__sl3)
        >>> solution = solutions[2]

        Turn into cross Ratios:

        >>> crossRatios = solution.cross_ratios()

        Get a cross ratio:

        >>> crossRatios['zp_0010_0']
        Mod(-x, x^2 + x + 1)

        Check the relationship between cross ratios:

        >>> crossRatios['z_0010_0'] == 1 - 1 / crossRatios['zp_0010_0']
        True

        >>> crossRatios['zp_0010_0'] == 1 - 1 / crossRatios['zpp_0010_0']
        True

        >>> crossRatios['zpp_0010_0'] == 1 - 1 / crossRatios['z_0010_0']
        True

        Get information about what one can do with cross ratios
        """
        return CrossRatios(_ptolemy_to_cross_ratio(self)[0], is_numerical=self._is_numerical, manifold_thunk=self._manifold_thunk)

    def cross_ratios_numerical(self):
        """
        Turn exact solutions into numerical and then compute cross ratios.
        See numerical() and cross_ratios().
        """
        if self._is_numerical:
            return self.cross_ratios()
        else:
            return ZeroDimensionalComponent([num.cross_ratios() for num in self.numerical()])

    def flattenings_numerical(self):
        """
        Turn into numerical solutions and compute flattenings, see
        help(snappy.ptolemy.coordinates.Flattenings)
        Also see numerical()

        Get Ptolemy coordinates.

        >>> from snappy.ptolemy.processMagmaFile import _magma_output_for_4_1__sl3, solutions_from_magma
        >>> solutions = solutions_from_magma(_magma_output_for_4_1__sl3)
        >>> solution = solutions[2]

        Compute a numerical solution

        >>> flattenings = solution.flattenings_numerical()

        Get more information with help(flattenings[0])
        """
        if self._is_numerical:
            branch_factor = 1
            for i in range(1000):
                try:
                    d, evenN = _ptolemy_to_cross_ratio(self, branch_factor, self._non_trivial_generalized_obstruction_class, as_flattenings=True)
                    return Flattenings(d, manifold_thunk=self._manifold_thunk, evenN=evenN)
                except LogToCloseToBranchCutError:
                    branch_factor *= pari('exp(0.0001 * I)')
            raise Exception('Could not find non-ambiguous branch cut for log')
        else:
            return ZeroDimensionalComponent([num.flattenings_numerical() for num in self.numerical()])

    def volume_numerical(self, drop_negative_vols=False):
        """
        Turn into (Galois conjugate) numerical solutions and compute volumes.
        If already numerical, only return the one volume.
        See numerical().

        If drop_negative_vols = True is given as optional argument,
        only return non-negative volumes.
        """
        if self._is_numerical:
            return self.cross_ratios().volume_numerical()
        else:
            vols = ZeroDimensionalComponent([num.volume_numerical() for num in self.numerical()])
            if drop_negative_vols:
                return [vol for vol in vols if vol > -1e-12]
            return vols

    def complex_volume_numerical(self, drop_negative_vols=False, with_modulo=False):
        """
        Turn into (Galois conjugate) numerical solutions and compute complex
        volumes. If already numerical, return the volume.

        Complex volume is defined up to i*pi**2/6.

        See numerical(). If drop_negative_vols = True is given as optional
        argument, only return complex volumes with non-negative real part.
        """
        if self._is_numerical:
            return self.flattenings_numerical().complex_volume(with_modulo=with_modulo)
        else:
            cvols = ZeroDimensionalComponent([num.flattenings_numerical().complex_volume(with_modulo=with_modulo) for num in self.numerical()])
            if drop_negative_vols:
                return [cvol for cvol in cvols if cvol.real() > -1e-12]
            return cvols

    def _coordinate_at_tet_and_point(self, tet, pt):
        """
        Given the index of a tetrahedron and a quadruple (any iterable) of
        integer to mark an integral point on that tetrahedron, returns the
        associated Ptolemy coordinate.
        If this is a vertex Ptolemy coordinate, always return 1 without
        checking for it in the dictionary.
        """
        if sum(pt) in pt:
            return 1
        return self['c_%d%d%d%d' % tuple(pt) + '_%d' % tet]

    def _get_obstruction_variable(self, face, tet):
        """
        Get the obstruction variable sigma_face for the given face and
        tetrahedron. Return 1 if there is no such obstruction class.
        """
        key = 's_%d_%d' % (face, tet)
        return self.get(key, +1)

    @staticmethod
    def _three_perm_sign(v0, v1, v2):
        """
        Returns the sign of the permutation necessary to bring the three
        elements v0, v1, v2 in order.
        """
        if v0 < v2 and v2 < v1:
            return -1
        if v1 < v0 and v0 < v2:
            return -1
        if v2 < v1 and v1 < v0:
            return -1
        return +1

    def diamond_coordinate(self, tet, v0, v1, v2, pt):
        """
        Returns the diamond coordinate for tetrahedron with index tet
        for the face with vertices v0, v1, v2 (integers between 0 and 3) and
        integral point pt (quadruple adding up to N-2).

        See Definition 10.1:
        Garoufalidis, Goerner, Zickert:
        Gluing Equations for PGL(n,C)-Representations of 3-Manifolds
        http://arxiv.org/abs/1207.6711
        """
        pt_v0_v0 = [a + 2 * _kronecker_delta(v0, i) for i, a in enumerate(pt)]
        pt_v0_v1 = [a + _kronecker_delta(v0, i) + _kronecker_delta(v1, i) for i, a in enumerate(pt)]
        pt_v0_v2 = [a + _kronecker_delta(v0, i) + _kronecker_delta(v2, i) for i, a in enumerate(pt)]
        pt_v1_v2 = [a + _kronecker_delta(v1, i) + _kronecker_delta(v2, i) for i, a in enumerate(pt)]
        c_pt_v0_v0 = self._coordinate_at_tet_and_point(tet, pt_v0_v0)
        c_pt_v0_v1 = self._coordinate_at_tet_and_point(tet, pt_v0_v1)
        c_pt_v0_v2 = self._coordinate_at_tet_and_point(tet, pt_v0_v2)
        c_pt_v1_v2 = self._coordinate_at_tet_and_point(tet, pt_v1_v2)
        face = list(set(range(4)) - set([v0, v1, v2]))[0]
        obstruction = self._get_obstruction_variable(face, tet)
        s = PtolemyCoordinates._three_perm_sign(v0, v1, v2)
        return -(obstruction * s * (c_pt_v0_v0 * c_pt_v1_v2) / (c_pt_v0_v1 * c_pt_v0_v2))

    def ratio_coordinate(self, tet, v0, v1, pt):
        """
        Returns the ratio coordinate for tetrahedron with index tet
        for the edge from v0 to v1 (integers between 0 and 3) and integral
        point pt (quadruple adding up N-1) on the edge.

        See Definition 10.2:
        Garoufalidis, Goerner, Zickert:
        Gluing Equations for PGL(n,C)-Representations of 3-Manifolds
        http://arxiv.org/abs/1207.6711

        Note that this definition turned out to have the wrong sign. Multiply
        the result by -1 if v1 < v0 and N is even.
        """
        pt_v0 = [a + _kronecker_delta(v0, i) for i, a in enumerate(pt)]
        pt_v1 = [a + _kronecker_delta(v1, i) for i, a in enumerate(pt)]
        c_pt_v0 = self._coordinate_at_tet_and_point(tet, pt_v0)
        c_pt_v1 = self._coordinate_at_tet_and_point(tet, pt_v1)
        s = (-1) ** pt[v1]
        if v1 < v0 and self.N() % 2 == 0:
            s *= -1
        return s * c_pt_v1 / c_pt_v0

    def _get_identity_matrix(self):
        N = self.N()
        return [[_kronecker_delta(i, j) for i in range(N)] for j in range(N)]

    def long_edge(self, tet, v0, v1, v2):
        """
        The matrix that labels a long edge from v0 to v1 (integers between 0
        and 3) of a (doubly) truncated simplex corresponding to an ideal
        tetrahedron with index tet.

        This matrix was labeled alpha^{v0v1v2} (v2 does not matter for non
        double-truncated simplex) in Figure 18 of
        Garoufalidis, Goerner, Zickert:
        Gluing Equations for PGL(n,C)-Representations of 3-Manifolds
        http://arxiv.org/abs/1207.6711

        It is computed using equation 10.4. Note that the ratio coordinate
        is different from the definition in the paper (see ratio_coordinate).

        The resulting matrix is given as a python list of lists.
        """
        key = 'long_%d_%d%d' % (tet, v0, v1)
        if key not in self._edge_cache:
            N = self.N()
            m = [[0 for i in range(N)] for j in range(N)]
            for c in range(N):
                r = N - 1 - c
                pt = [r * _kronecker_delta(v0, i) + c * _kronecker_delta(v1, i) for i in range(4)]
                m[r][c] = self.ratio_coordinate(tet, v0, v1, pt)
            self._edge_cache[key] = m
        return self._edge_cache[key]

    def middle_edge(self, tet, v0, v1, v2):
        """
        The matrix that labels a middle edge on the face v0, v1, v2 (integers
        between 0 and 3) of a doubly truncated simplex (or a short edge of the
        truncated simplex) corresponding to an ideal tetrahedron with index
        tet.

        This matrix was labeled beta^{v0v1v2} in Figure 18 of
        Garoufalidis, Goerner, Zickert:
        Gluing Equations for PGL(n,C)-Representations of 3-Manifolds
        http://arxiv.org/abs/1207.6711

        It is computed using equation 10.4.

        The resulting matrix is given as a python list of lists.
        """
        key = 'middle_%d_%d%d%d' % (tet, v0, v1, v2)
        if key not in self._edge_cache:
            N = self.N()
            m = self._get_identity_matrix()
            for a0, a1, a2 in utilities.triples_with_fixed_sum_iterator(N - 2):
                pt = [a1 * _kronecker_delta(v0, i) + a2 * _kronecker_delta(v1, i) + a0 * _kronecker_delta(v2, i) for i in range(4)]
                diamond = self.diamond_coordinate(tet, v0, v1, v2, pt)
                m = matrix.matrix_mult(m, _X(N, a1 + 1, diamond))
            self._edge_cache[key] = m
        return self._edge_cache[key]

    def short_edge(self, tet, v0, v1, v2):
        """
        Returns the identity. This is because we can think of the matrices
        given by Ptolemy coordinates of living on truncated simplices which
        can be though of as doubly truncated simplices where all short edges
        are collapsed, hence labeled by the identity.

        See equation 10.4 in
        Garoufalidis, Goerner, Zickert:
        Gluing Equations for PGL(n,C)-Representations of 3-Manifolds
        http://arxiv.org/abs/1207.6711
        """
        key = 'short'
        if key not in self._edge_cache:
            N = self.N()
            m = self._get_identity_matrix()
            self._edge_cache[key] = m
        return self._edge_cache[key]

    def _init_matrix_and_inverse_cache(self):
        if self._matrix_cache and self._inverse_matrix_cache:
            return
        self._matrix_cache, self._inverse_matrix_cache = findLoops.images_of_original_generators(self, penalties=(0, 1, 1))

    def evaluate_word(self, word, G=None):
        """
        Given a word in the generators of the fundamental group,
        compute the corresponding matrix. By default, these are the
        generators of the unsimplified presentation of the fundamental
        group. An optional SnapPy fundamental group can be given if the
        words are in generators of a different presentation, e.g.,
        c.evaluate_word(word, M.fundamental_group(True)) to
        evaluate a word in the simplified presentation returned by
        M.fundamental_group(True).

        For now, the matrix is returned as list of lists.
        """
        self._init_matrix_and_inverse_cache()
        return findLoops.evaluate_word(self._get_identity_matrix(), self._matrix_cache, self._inverse_matrix_cache, word, G)

    def _testing_assert_identity(self, m, allow_sign_if_obstruction_class=False):
        N = self.N()
        null = [[0 for i in range(N)] for j in range(N)]
        identity = self._get_identity_matrix()
        if allow_sign_if_obstruction_class and self.has_obstruction():
            if not (matrix.matrix_add(m, identity) == null or matrix.matrix_sub(m, identity) == null):
                raise Exception('Cocycle condition violated: %s' % m)
        elif not matrix.matrix_sub(m, identity) == null:
            raise Exception('Cocycle condition violated: %s' % m)

    def _testing_check_cocycles(self):
        for tet in range(self.num_tetrahedra()):
            for v in [(0, 1, 2), (0, 1, 3), (0, 2, 1), (0, 2, 3), (0, 3, 1), (0, 3, 2), (1, 0, 2), (1, 0, 3), (1, 2, 0), (1, 2, 3), (1, 3, 0), (1, 3, 2), (2, 0, 1), (2, 0, 3), (2, 1, 0), (2, 1, 3), (2, 3, 0), (2, 3, 1), (3, 0, 1), (3, 0, 2), (3, 1, 0), (3, 1, 2), (3, 2, 0), (3, 2, 1)]:
                m1 = self.middle_edge(tet, v[0], v[1], v[2])
                m2 = self.middle_edge(tet, v[0], v[2], v[1])
                self._testing_assert_identity(matrix.matrix_mult(m1, m2))
            for v in [(0, 1, 2), (0, 2, 1), (0, 3, 1), (1, 0, 2), (1, 2, 0), (1, 3, 0), (2, 0, 1), (2, 1, 0), (2, 3, 0), (3, 0, 1), (3, 1, 0), (3, 2, 0)]:
                m1 = self.long_edge(tet, v[0], v[1], v[2])
                m2 = self.long_edge(tet, v[1], v[0], v[2])
                self._testing_assert_identity(matrix.matrix_mult(m1, m2))
            for v in [(0, 1, 2, 3), (1, 2, 3, 0), (2, 3, 0, 1), (3, 0, 1, 2)]:
                m1 = self.middle_edge(tet, v[0], v[1], v[2])
                m2 = self.middle_edge(tet, v[0], v[2], v[3])
                m3 = self.middle_edge(tet, v[0], v[3], v[1])
                self._testing_assert_identity(matrix.matrix_mult(m1, matrix.matrix_mult(m2, m3)))
            for v in [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]:
                m1 = self.middle_edge(tet, v[0], v[1], v[2])
                m2 = self.long_edge(tet, v[0], v[2], v[1])
                m3 = self.middle_edge(tet, v[2], v[0], v[1])
                m4 = self.long_edge(tet, v[2], v[1], v[0])
                m5 = self.middle_edge(tet, v[1], v[2], v[0])
                m6 = self.long_edge(tet, v[1], v[0], v[2])
                self._testing_assert_identity(matrix.matrix_mult(m1, matrix.matrix_mult(m2, matrix.matrix_mult(m3, matrix.matrix_mult(m4, matrix.matrix_mult(m5, m6))))), True)

    def check_against_manifold(self, M=None, epsilon=None):
        """
        Checks that the given solution really is a solution to the Ptolemy
        variety of a manifold. See help(ptolemy.PtolemyCoordinates) for
        more example.

        === Arguments ===

        * M --- manifold to check this for
        * epsilon --- maximal allowed error when checking the relations, use
          None for exact comparison.
        """
        if M is None:
            M = self.get_manifold()
        if M is None:
            raise Exception('Need to give manifold')
        if self._non_trivial_generalized_obstruction_class:
            raise PtolemyCannotBeCheckedError()
        num_tets = self.num_tetrahedra()
        N, has_obstruction_class = _N_and_has_obstruction_for_ptolemys(self)
        if not M.num_tetrahedra() == num_tets:
            raise Exception('Number tetrahedra not matching')
        if has_obstruction_class:
            for tet in range(num_tets):
                _check_relation(self._get_obstruction_variable(0, tet) * self._get_obstruction_variable(1, tet) * self._get_obstruction_variable(2, tet) * self._get_obstruction_variable(3, tet) - 1, epsilon, 'Obstruction cocycle')
            for dummy_sign, power, var1, var2 in M._ptolemy_equations_identified_face_classes():
                _check_relation(self[var1] - self[var2], epsilon, 'Identification of face classes')
        for sign, power, var1, var2 in M._ptolemy_equations_identified_coordinates(N):
            _check_relation(self[var1] - sign * self[var2], epsilon, 'Identification of Ptolemy coordinates')
        for tet in range(num_tets):
            for index in utilities.quadruples_with_fixed_sum_iterator(N - 2):

                def get_ptolemy_coordinate(addl_index):
                    total_index = matrix.vector_add(index, addl_index)
                    key = 'c_%d%d%d%d' % tuple(total_index) + '_%d' % tet
                    return self[key]
                s0 = self._get_obstruction_variable(0, tet)
                s1 = self._get_obstruction_variable(1, tet)
                s2 = self._get_obstruction_variable(2, tet)
                s3 = self._get_obstruction_variable(3, tet)
                rel = s0 * s1 * get_ptolemy_coordinate((1, 1, 0, 0)) * get_ptolemy_coordinate((0, 0, 1, 1)) - s0 * s2 * get_ptolemy_coordinate((1, 0, 1, 0)) * get_ptolemy_coordinate((0, 1, 0, 1)) + s0 * s3 * get_ptolemy_coordinate((1, 0, 0, 1)) * get_ptolemy_coordinate((0, 1, 1, 0))
                _check_relation(rel, epsilon, 'Ptolemy relation')

    def is_geometric(self, epsilon=1e-06):
        """
        Returns true if all shapes corresponding to this solution have positive
        imaginary part.

        If the solutions are exact, it returns true if one of the corresponding
        numerical solutions is geometric.

        An optional epsilon can be given. An imaginary part of a shape is
        considered positive if it is larger than this epsilon.
        """
        if self._is_numerical:
            return self.cross_ratios().is_geometric(epsilon)
        else:
            for numerical_sol in self.numerical():
                if numerical_sol.is_geometric(epsilon):
                    return True
            return False