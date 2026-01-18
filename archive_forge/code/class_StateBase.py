from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import Function
from sympy.core.numbers import oo, equal_valued
from sympy.core.singleton import S
from sympy.functions.elementary.complexes import conjugate
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.integrals.integrals import integrate
from sympy.printing.pretty.stringpict import stringPict
from sympy.physics.quantum.qexpr import QExpr, dispatch_method
class StateBase(QExpr):
    """Abstract base class for general abstract states in quantum mechanics.

    All other state classes defined will need to inherit from this class. It
    carries the basic structure for all other states such as dual, _eval_adjoint
    and label.

    This is an abstract base class and you should not instantiate it directly,
    instead use State.
    """

    @classmethod
    def _operators_to_state(self, ops, **options):
        """ Returns the eigenstate instance for the passed operators.

        This method should be overridden in subclasses. It will handle being
        passed either an Operator instance or set of Operator instances. It
        should return the corresponding state INSTANCE or simply raise a
        NotImplementedError. See cartesian.py for an example.
        """
        raise NotImplementedError('Cannot map operators to states in this class. Method not implemented!')

    def _state_to_operators(self, op_classes, **options):
        """ Returns the operators which this state instance is an eigenstate
        of.

        This method should be overridden in subclasses. It will be called on
        state instances and be passed the operator classes that we wish to make
        into instances. The state instance will then transform the classes
        appropriately, or raise a NotImplementedError if it cannot return
        operator instances. See cartesian.py for examples,
        """
        raise NotImplementedError('Cannot map this state to operators. Method not implemented!')

    @property
    def operators(self):
        """Return the operator(s) that this state is an eigenstate of"""
        from .operatorset import state_to_operators
        return state_to_operators(self)

    def _enumerate_state(self, num_states, **options):
        raise NotImplementedError('Cannot enumerate this state!')

    def _represent_default_basis(self, **options):
        return self._represent(basis=self.operators)

    @property
    def dual(self):
        """Return the dual state of this one."""
        return self.dual_class()._new_rawargs(self.hilbert_space, *self.args)

    @classmethod
    def dual_class(self):
        """Return the class used to construct the dual."""
        raise NotImplementedError('dual_class must be implemented in a subclass')

    def _eval_adjoint(self):
        """Compute the dagger of this state using the dual."""
        return self.dual

    def _pretty_brackets(self, height, use_unicode=True):
        if use_unicode:
            lbracket, rbracket = (getattr(self, 'lbracket_ucode', ''), getattr(self, 'rbracket_ucode', ''))
            slash, bslash, vert = ('╱', '╲', '│')
        else:
            lbracket, rbracket = (getattr(self, 'lbracket', ''), getattr(self, 'rbracket', ''))
            slash, bslash, vert = ('/', '\\', '|')
        if height == 1:
            return (stringPict(lbracket), stringPict(rbracket))
        height += height % 2
        brackets = []
        for bracket in (lbracket, rbracket):
            if bracket in {_lbracket, _lbracket_ucode}:
                bracket_args = [' ' * (height // 2 - i - 1) + slash for i in range(height // 2)]
                bracket_args.extend([' ' * i + bslash for i in range(height // 2)])
            elif bracket in {_rbracket, _rbracket_ucode}:
                bracket_args = [' ' * i + bslash for i in range(height // 2)]
                bracket_args.extend([' ' * (height // 2 - i - 1) + slash for i in range(height // 2)])
            elif bracket in {_straight_bracket, _straight_bracket_ucode}:
                bracket_args = [vert] * height
            else:
                raise ValueError(bracket)
            brackets.append(stringPict('\n'.join(bracket_args), baseline=height // 2))
        return brackets

    def _sympystr(self, printer, *args):
        contents = self._print_contents(printer, *args)
        return '%s%s%s' % (getattr(self, 'lbracket', ''), contents, getattr(self, 'rbracket', ''))

    def _pretty(self, printer, *args):
        from sympy.printing.pretty.stringpict import prettyForm
        pform = self._print_contents_pretty(printer, *args)
        lbracket, rbracket = self._pretty_brackets(pform.height(), printer._use_unicode)
        pform = prettyForm(*pform.left(lbracket))
        pform = prettyForm(*pform.right(rbracket))
        return pform

    def _latex(self, printer, *args):
        contents = self._print_contents_latex(printer, *args)
        return '{%s%s%s}' % (getattr(self, 'lbracket_latex', ''), contents, getattr(self, 'rbracket_latex', ''))