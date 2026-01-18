from pyomo.common.deprecation import deprecated
from pyomo.common.modeling import NOTSET
import pyomo.core.expr as EXPR
from pyomo.core.kernel.base import ICategorizedObject, _abstract_readwrite_property
from pyomo.core.kernel.container_utils import define_simple_containers
from pyomo.core.expr.numvalue import (
class IIdentityExpression(NumericValue):
    """The interface for classes that simply wrap another
    expression and perform no additional operations.

    Derived classes should declare an _expr attribute or
    override all implemented methods.
    """
    __slots__ = ()
    PRECEDENCE = 0
    ASSOCIATIVITY = EXPR.OperatorAssociativity.NON_ASSOCIATIVE

    @property
    def expr(self):
        return self._expr

    def __call__(self, exception=False):
        """Compute the value of this expression.

        Args:
            exception (bool): Indicates if an exception
                should be raised when instances of
                NumericValue fail to evaluate due to one or
                more objects not being initialized to a
                numeric value (e.g, one or more variables in
                an algebraic expression having the value
                None). Default is :const:`True`.

        Returns:
            numeric value or None
        """
        return value(self._expr, exception=exception)

    def is_fixed(self):
        """A boolean indicating whether this expression is fixed."""
        return is_fixed(self._expr)

    def is_parameter_type(self):
        """A boolean indicating whether this expression is a parameter object."""
        return False

    def is_variable_type(self):
        """A boolean indicating whether this expression is a
        variable object."""
        return False

    def is_named_expression_type(self):
        """A boolean indicating whether this in a named expression."""
        return True

    def is_expression_type(self, expression_system=None):
        """A boolean indicating whether this in an expression."""
        return True

    @property
    def _args_(self):
        """A tuple of subexpressions involved in this expressions operation."""
        return (self._expr,)

    @property
    def args(self):
        """A tuple of subexpressions involved in this expressions operation."""
        yield self._expr

    def nargs(self):
        """Length of self._nargs()"""
        return 1

    def arg(self, i):
        if i != 0:
            raise KeyError('Unexpected argument id')
        return self._expr

    def polynomial_degree(self):
        """The polynomial degree of the stored expression."""
        if self.is_fixed():
            return 0
        return self._expr.polynomial_degree()

    def _compute_polynomial_degree(self, values):
        return values[0]

    def to_string(self, verbose=None, labeler=None, smap=None, compute_values=False):
        """Convert this expression into a string."""
        return EXPR.expression_to_string(self, verbose=verbose, labeler=labeler, smap=smap, compute_values=compute_values)

    def _to_string(self, values, verbose, smap):
        if verbose:
            name = self.getname()
            if name == None:
                return '<%s>{%s}' % (self.__class__.__name__, values[0])
            else:
                if name[0] == '<':
                    name = ''
                return '%s{%s}' % (name, values[0])
        if self._expr is None:
            return '%s{Undefined}' % str(self)
        return values[0]

    def _apply_operation(self, result):
        return result[0]

    def _is_fixed(self, values):
        return values[0]

    def create_node_with_local_data(self, values):
        """
        Construct an expression after constructing the
        contained expression.

        This class provides a consistent interface for constructing a
        node, which is used in tree visitor scripts.
        """
        return self.__class__(expr=values[0])

    def is_constant(self):
        raise NotImplementedError

    def is_potentially_variable(self):
        raise NotImplementedError

    def clone(self):
        raise NotImplementedError