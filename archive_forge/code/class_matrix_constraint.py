from pyomo.common.dependencies import (
from pyomo.core.expr.numvalue import NumericValue, value
from pyomo.core.kernel.constraint import IConstraint, constraint_tuple
class matrix_constraint(constraint_tuple):
    """
    A container for constraints of the form lb <= Ax <= ub.

    Args:
        A: A scipy sparse matrix or 2D numpy array (always
            copied)
        lb: A scalar or array with the same number of rows
            as A that defines the lower bound of the
            constraints
        ub: A scalar or array with the same number of rows
            as A that defines the upper bound of the
            constraints
        rhs: A scalar or array with the same number of rows
            as A that defines the right-hand side of the
            constraints (implies equality constraints)
        x: A list with the same number of columns as A that
            stores the variable associated with each column
        sparse: Indicates whether or not sparse storage (CSR
            format) should be used to store A. Default is
            :const:`True`.
    """
    __slots__ = ('_A', '_sparse', '_lb', '_ub', '_equality', '_x')

    def __init__(self, A, lb=None, ub=None, rhs=None, x=None, sparse=True):
        if not has_numpy or not has_scipy:
            raise ValueError('This class requires numpy and scipy')
        m, n = A.shape
        assert m > 0
        assert n > 0
        cons = (_MatrixConstraintData(i) for i in range(m))
        super(matrix_constraint, self).__init__(cons)
        if sparse:
            self._sparse = True
            self._A = scipy.sparse.csr_matrix(A, dtype=float, copy=True)
            self._A.data.setflags(write=False)
            self._A.indices.setflags(write=False)
            self._A.indptr.setflags(write=False)
        else:
            self._sparse = False
            self._A = numpy.array(A, dtype=float, copy=True)
            self._A.setflags(write=False)
        self._lb = numpy.ndarray(m, dtype=float)
        self._ub = numpy.ndarray(m, dtype=float)
        self._equality = numpy.ndarray(m, dtype=bool)
        self._equality.fill(False)
        self.x = x
        if rhs is None:
            self.lb = lb
            self.ub = ub
        else:
            if lb is not None or ub is not None:
                raise ValueError("The 'rhs' keyword can not be used with the 'lb' or 'ub' keywords to initialize a constraint.")
            self.rhs = rhs

    @property
    def sparse(self):
        """Boolean indicating whether or not the underlying
        matrix uses sparse storage"""
        return self._sparse

    @property
    def A(self):
        """A read-only view of the constraint matrix"""
        if self._sparse:
            return scipy.sparse.csr_matrix(self._A, copy=False)
        else:
            return self._A.view()

    @property
    def x(self):
        """The list of variables associated with the columns
        of the constraint matrix"""
        return self._x

    @x.setter
    def x(self, x):
        if x is None:
            self._x = None
        else:
            x = tuple(x)
            m, n = self._A.shape
            if len(x) != n:
                raise ValueError('Argument length must be %s not %s' % (n, len(x)))
            self._x = x

    @property
    def lb(self):
        """The array of constraint lower bounds"""
        return self._lb.view()

    @lb.setter
    def lb(self, lb):
        if self.equality.any():
            raise ValueError('The lb array can not be set when there are indices of the equality array that are True')
        if lb is None:
            lb = -numpy.inf
        if isinstance(lb, numpy.ndarray):
            numpy.copyto(self._lb, lb)
        elif isinstance(lb, NumericValue):
            raise ValueError('lb must be set to a simple numeric type or a numpy array')
        else:
            self._lb.fill(lb)

    @property
    def ub(self):
        """The array of constraint upper bounds"""
        return self._ub.view()

    @ub.setter
    def ub(self, ub):
        if self.equality.any():
            raise ValueError('The ub array can not be set when there are indices of the equality array that are True')
        if ub is None:
            ub = numpy.inf
        if isinstance(ub, numpy.ndarray):
            numpy.copyto(self._ub, ub)
        elif isinstance(ub, NumericValue):
            raise ValueError('ub must be set to a simple numeric type or a numpy array')
        else:
            self._ub.fill(ub)

    @property
    def rhs(self):
        """The array of constraint right-hand sides. Can be
        set to a scalar or a numpy array of the same
        dimension. This property can only be read when the
        equality property is :const:`True` on every
        index. Assigning to this property implicitly sets
        the equality property to :const:`True` on every
        index."""
        if not self.equality.all():
            raise ValueError('The rhs array can not be read when there are indices of the equality array that are False.')
        return self._lb.view()

    @rhs.setter
    def rhs(self, rhs):
        if rhs is None:
            raise ValueError('Constraint right-hand side can not be assigned a value of None.')
        elif isinstance(rhs, NumericValue):
            raise ValueError('rhs must be set to a simple numeric type or a numpy array')
        elif isinstance(rhs, numpy.ndarray):
            numpy.copyto(self._lb, rhs)
            numpy.copyto(self._ub, rhs)
        else:
            self._lb.fill(rhs)
            self._ub.fill(rhs)
        self._equality.fill(True)

    @property
    def equality(self):
        """The array of boolean entries indicating the
        indices that are equality constraints"""
        return self._equality.view()

    @equality.setter
    def equality(self, equality):
        if equality:
            raise ValueError('The constraint equality flag can only be set to True by assigning an expression to the rhs property (e.g., con.rhs = con.lb).')
        assert not equality
        self._equality.fill(False)

    def __call__(self, exception=True):
        """Compute the value of the body of this constraint"""
        if self.x is None:
            raise ValueError('No variable order has been assigned')
        values = numpy.array([v.value for v in self.x], dtype=float)
        if numpy.isnan(values).any():
            if exception:
                raise ValueError('One or more variables do not have a value')
            return None
        return self._A.dot(values)

    @property
    def lslack(self, body=_noarg):
        """Lower slack (body - lb)"""
        if body is _noarg:
            body = self(exception=False)
        if body is None:
            return None
        return body - self.lb

    @property
    def uslack(self, body=_noarg):
        """Upper slack (ub - body)"""
        if body is _noarg:
            body = self(exception=False)
        if body is None:
            return None
        return self.ub - body

    @property
    def slack(self):
        """min(lslack, uslack)"""
        body = self(exception=False)
        if body is None:
            return None
        lslack = self.__class__.lslack.fget(self, body=body)
        uslack = self.__class__.uslack.fget(self, body=body)
        return numpy.minimum(lslack, uslack)