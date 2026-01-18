from sympy.core.numbers import Integer
from sympy.core.singleton import S
from sympy.functions.combinatorial.factorials import factorial

    Calculate an approximation for lim k->oo A(k) using the n-term Shanks
    transformation S(A)(n). With m > 1, calculate the m-fold recursive
    Shanks transformation S(S(...S(A)...))(n).

    The Shanks transformation is useful for summing Taylor series that
    converge slowly near a pole or singularity, e.g. for log(2):

        >>> from sympy.abc import k, n
        >>> from sympy import Sum, Integer
        >>> from sympy.series.acceleration import shanks
        >>> A = Sum(Integer(-1)**(k+1) / k, (k, 1, n))
        >>> print(round(A.subs(n, 100).doit().evalf(), 10))
        0.6881721793
        >>> print(round(shanks(A, n, 25).evalf(), 10))
        0.6931396564
        >>> print(round(shanks(A, n, 25, 5).evalf(), 10))
        0.6931471806

    The correct value is 0.6931471805599453094172321215.
    