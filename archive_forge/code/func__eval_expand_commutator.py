from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.printing.pretty.stringpict import prettyForm
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.operator import Operator
def _eval_expand_commutator(self, **hints):
    A = self.args[0]
    B = self.args[1]
    if isinstance(A, Add):
        sargs = []
        for term in A.args:
            comm = Commutator(term, B)
            if isinstance(comm, Commutator):
                comm = comm._eval_expand_commutator()
            sargs.append(comm)
        return Add(*sargs)
    elif isinstance(B, Add):
        sargs = []
        for term in B.args:
            comm = Commutator(A, term)
            if isinstance(comm, Commutator):
                comm = comm._eval_expand_commutator()
            sargs.append(comm)
        return Add(*sargs)
    elif isinstance(A, Mul):
        a = A.args[0]
        b = Mul(*A.args[1:])
        c = B
        comm1 = Commutator(b, c)
        comm2 = Commutator(a, c)
        if isinstance(comm1, Commutator):
            comm1 = comm1._eval_expand_commutator()
        if isinstance(comm2, Commutator):
            comm2 = comm2._eval_expand_commutator()
        first = Mul(a, comm1)
        second = Mul(comm2, b)
        return Add(first, second)
    elif isinstance(B, Mul):
        a = A
        b = B.args[0]
        c = Mul(*B.args[1:])
        comm1 = Commutator(a, b)
        comm2 = Commutator(a, c)
        if isinstance(comm1, Commutator):
            comm1 = comm1._eval_expand_commutator()
        if isinstance(comm2, Commutator):
            comm2 = comm2._eval_expand_commutator()
        first = Mul(comm1, c)
        second = Mul(b, comm2)
        return Add(first, second)
    elif isinstance(A, Pow):
        return self._expand_pow(A, B, 1)
    elif isinstance(B, Pow):
        return self._expand_pow(B, A, -1)
    return self