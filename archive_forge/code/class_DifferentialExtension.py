from the names used in Bronstein's book.
from types import GeneratorType
from functools import reduce
from sympy.core.function import Lambda
from sympy.core.mul import Mul
from sympy.core.numbers import ilcm, I, oo
from sympy.core.power import Pow
from sympy.core.relational import Ne
from sympy.core.singleton import S
from sympy.core.sorting import ordered, default_sort_key
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.elementary.exponential import log, exp
from sympy.functions.elementary.hyperbolic import (cosh, coth, sinh,
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (atan, sin, cos,
from .integrals import integrate, Integral
from .heurisch import _symbols
from sympy.polys.polyerrors import DomainError, PolynomialError
from sympy.polys.polytools import (real_roots, cancel, Poly, gcd,
from sympy.polys.rootoftools import RootSum
from sympy.utilities.iterables import numbered_symbols
class DifferentialExtension:
    """
    A container for all the information relating to a differential extension.

    Explanation
    ===========

    The attributes of this object are (see also the docstring of __init__):

    - f: The original (Expr) integrand.
    - x: The variable of integration.
    - T: List of variables in the extension.
    - D: List of derivations in the extension; corresponds to the elements of T.
    - fa: Poly of the numerator of the integrand.
    - fd: Poly of the denominator of the integrand.
    - Tfuncs: Lambda() representations of each element of T (except for x).
      For back-substitution after integration.
    - backsubs: A (possibly empty) list of further substitutions to be made on
      the final integral to make it look more like the integrand.
    - exts:
    - extargs:
    - cases: List of string representations of the cases of T.
    - t: The top level extension variable, as defined by the current level
      (see level below).
    - d: The top level extension derivation, as defined by the current
      derivation (see level below).
    - case: The string representation of the case of self.d.
    (Note that self.T and self.D will always contain the complete extension,
    regardless of the level.  Therefore, you should ALWAYS use DE.t and DE.d
    instead of DE.T[-1] and DE.D[-1].  If you want to have a list of the
    derivations or variables only up to the current level, use
    DE.D[:len(DE.D) + DE.level + 1] and DE.T[:len(DE.T) + DE.level + 1].  Note
    that, in particular, the derivation() function does this.)

    The following are also attributes, but will probably not be useful other
    than in internal use:
    - newf: Expr form of fa/fd.
    - level: The number (between -1 and -len(self.T)) such that
      self.T[self.level] == self.t and self.D[self.level] == self.d.
      Use the methods self.increment_level() and self.decrement_level() to change
      the current level.
    """
    __slots__ = ('f', 'x', 'T', 'D', 'fa', 'fd', 'Tfuncs', 'backsubs', 'exts', 'extargs', 'cases', 'case', 't', 'd', 'newf', 'level', 'ts', 'dummy')

    def __init__(self, f=None, x=None, handle_first='log', dummy=False, extension=None, rewrite_complex=None):
        """
        Tries to build a transcendental extension tower from ``f`` with respect to ``x``.

        Explanation
        ===========

        If it is successful, creates a DifferentialExtension object with, among
        others, the attributes fa, fd, D, T, Tfuncs, and backsubs such that
        fa and fd are Polys in T[-1] with rational coefficients in T[:-1],
        fa/fd == f, and D[i] is a Poly in T[i] with rational coefficients in
        T[:i] representing the derivative of T[i] for each i from 1 to len(T).
        Tfuncs is a list of Lambda objects for back replacing the functions
        after integrating.  Lambda() is only used (instead of lambda) to make
        them easier to test and debug. Note that Tfuncs corresponds to the
        elements of T, except for T[0] == x, but they should be back-substituted
        in reverse order.  backsubs is a (possibly empty) back-substitution list
        that should be applied on the completed integral to make it look more
        like the original integrand.

        If it is unsuccessful, it raises NotImplementedError.

        You can also create an object by manually setting the attributes as a
        dictionary to the extension keyword argument.  You must include at least
        D.  Warning, any attribute that is not given will be set to None. The
        attributes T, t, d, cases, case, x, and level are set automatically and
        do not need to be given.  The functions in the Risch Algorithm will NOT
        check to see if an attribute is None before using it.  This also does not
        check to see if the extension is valid (non-algebraic) or even if it is
        self-consistent.  Therefore, this should only be used for
        testing/debugging purposes.
        """
        if extension:
            if 'D' not in extension:
                raise ValueError('At least the key D must be included with the extension flag to DifferentialExtension.')
            for attr in extension:
                setattr(self, attr, extension[attr])
            self._auto_attrs()
            return
        elif f is None or x is None:
            raise ValueError('Either both f and x or a manual extension must be given.')
        if handle_first not in ('log', 'exp'):
            raise ValueError("handle_first must be 'log' or 'exp', not %s." % str(handle_first))
        self.f = f
        self.x = x
        self.dummy = dummy
        self.reset()
        exp_new_extension, log_new_extension = (True, True)
        if rewrite_complex is None:
            rewrite_complex = I in self.f.atoms()
        if rewrite_complex:
            rewritables = {(sin, cos, cot, tan, sinh, cosh, coth, tanh): exp, (asin, acos, acot, atan): log}
            for candidates, rule in rewritables.items():
                self.newf = self.newf.rewrite(candidates, rule)
            self.newf = cancel(self.newf)
        elif any((i.has(x) for i in self.f.atoms(sin, cos, tan, atan, asin, acos))):
            raise NotImplementedError('Trigonometric extensions are not supported (yet!)')
        exps = set()
        pows = set()
        numpows = set()
        sympows = set()
        logs = set()
        symlogs = set()
        while True:
            if self.newf.is_rational_function(*self.T):
                break
            if not exp_new_extension and (not log_new_extension):
                raise NotImplementedError("Couldn't find an elementary transcendental extension for %s.  Try using a " % str(f) + 'manual extension with the extension flag.')
            exps, pows, numpows, sympows, log_new_extension = self._rewrite_exps_pows(exps, pows, numpows, sympows, log_new_extension)
            logs, symlogs = self._rewrite_logs(logs, symlogs)
            if handle_first == 'exp' or not log_new_extension:
                exp_new_extension = self._exp_part(exps)
                if exp_new_extension is None:
                    self.f = self.newf
                    self.reset()
                    exp_new_extension = True
                    continue
            if handle_first == 'log' or not exp_new_extension:
                log_new_extension = self._log_part(logs)
        self.fa, self.fd = frac_in(self.newf, self.t)
        self._auto_attrs()
        return

    def __getattr__(self, attr):
        if attr not in self.__slots__:
            raise AttributeError('%s has no attribute %s' % (repr(self), repr(attr)))
        return None

    def _rewrite_exps_pows(self, exps, pows, numpows, sympows, log_new_extension):
        """
        Rewrite exps/pows for better processing.
        """
        from .prde import is_deriv_k
        ratpows = [i for i in self.newf.atoms(Pow) if isinstance(i.base, exp) and i.exp.is_Rational]
        ratpows_repl = [(i, i.base.base ** (i.exp * i.base.exp)) for i in ratpows]
        self.backsubs += [(j, i) for i, j in ratpows_repl]
        self.newf = self.newf.xreplace(dict(ratpows_repl))
        exps = update_sets(exps, self.newf.atoms(exp), lambda i: i.exp.is_rational_function(*self.T) and i.exp.has(*self.T))
        pows = update_sets(pows, self.newf.atoms(Pow), lambda i: i.exp.is_rational_function(*self.T) and i.exp.has(*self.T))
        numpows = update_sets(numpows, set(pows), lambda i: not i.base.has(*self.T))
        sympows = update_sets(sympows, set(pows) - set(numpows), lambda i: i.base.is_rational_function(*self.T) and (not i.exp.is_Integer))
        for i in ordered(pows):
            old = i
            new = exp(i.exp * log(i.base))
            if i in sympows:
                if i.exp.is_Rational:
                    raise NotImplementedError('Algebraic extensions are not supported (%s).' % str(i))
                basea, based = frac_in(i.base, self.t)
                A = is_deriv_k(basea, based, self)
                if A is None:
                    self.newf = self.newf.xreplace({old: new})
                    self.backsubs += [(new, old)]
                    log_new_extension = self._log_part([log(i.base)])
                    exps = update_sets(exps, self.newf.atoms(exp), lambda i: i.exp.is_rational_function(*self.T) and i.exp.has(*self.T))
                    continue
                ans, u, const = A
                newterm = exp(i.exp * (log(const) + u))
                self.newf = self.newf.xreplace({i: newterm})
            elif i not in numpows:
                continue
            else:
                newterm = new
            self.backsubs.append((new, old))
            self.newf = self.newf.xreplace({old: newterm})
            exps.append(newterm)
        return (exps, pows, numpows, sympows, log_new_extension)

    def _rewrite_logs(self, logs, symlogs):
        """
        Rewrite logs for better processing.
        """
        atoms = self.newf.atoms(log)
        logs = update_sets(logs, atoms, lambda i: i.args[0].is_rational_function(*self.T) and i.args[0].has(*self.T))
        symlogs = update_sets(symlogs, atoms, lambda i: i.has(*self.T) and i.args[0].is_Pow and i.args[0].base.is_rational_function(*self.T) and (not i.args[0].exp.is_Integer))
        for i in ordered(symlogs):
            lbase = log(i.args[0].base)
            logs.append(lbase)
            new = i.args[0].exp * lbase
            self.newf = self.newf.xreplace({i: new})
            self.backsubs.append((new, i))
        logs = sorted(set(logs), key=default_sort_key)
        return (logs, symlogs)

    def _auto_attrs(self):
        """
        Set attributes that are generated automatically.
        """
        if not self.T:
            self.T = [i.gen for i in self.D]
        if not self.x:
            self.x = self.T[0]
        self.cases = [get_case(d, t) for d, t in zip(self.D, self.T)]
        self.level = -1
        self.t = self.T[self.level]
        self.d = self.D[self.level]
        self.case = self.cases[self.level]

    def _exp_part(self, exps):
        """
        Try to build an exponential extension.

        Returns
        =======

        Returns True if there was a new extension, False if there was no new
        extension but it was able to rewrite the given exponentials in terms
        of the existing extension, and None if the entire extension building
        process should be restarted.  If the process fails because there is no
        way around an algebraic extension (e.g., exp(log(x)/2)), it will raise
        NotImplementedError.
        """
        from .prde import is_log_deriv_k_t_radical
        new_extension = False
        restart = False
        expargs = [i.exp for i in exps]
        ip = integer_powers(expargs)
        for arg, others in ip:
            others.sort(key=lambda i: i[1])
            arga, argd = frac_in(arg, self.t)
            A = is_log_deriv_k_t_radical(arga, argd, self)
            if A is not None:
                ans, u, n, const = A
                if n == -1:
                    n = 1
                    u **= -1
                    const *= -1
                    ans = [(i, -j) for i, j in ans]
                if n == 1:
                    self.newf = self.newf.xreplace({exp(arg): exp(const) * Mul(*[u ** power for u, power in ans])})
                    self.newf = self.newf.xreplace({exp(p * exparg): exp(const * p) * Mul(*[u ** power for u, power in ans]) for exparg, p in others})
                    continue
                elif const or len(ans) > 1:
                    rad = Mul(*[term ** (power / n) for term, power in ans])
                    self.newf = self.newf.xreplace({exp(p * exparg): exp(const * p) * rad for exparg, p in others})
                    self.newf = self.newf.xreplace(dict(list(zip(reversed(self.T), reversed([f(self.x) for f in self.Tfuncs])))))
                    restart = True
                    break
                else:
                    raise NotImplementedError('Cannot integrate over algebraic extensions.')
            else:
                arga, argd = frac_in(arg, self.t)
                darga = argd * derivation(Poly(arga, self.t), self) - arga * derivation(Poly(argd, self.t), self)
                dargd = argd ** 2
                darga, dargd = darga.cancel(dargd, include=True)
                darg = darga.as_expr() / dargd.as_expr()
                self.t = next(self.ts)
                self.T.append(self.t)
                self.extargs.append(arg)
                self.exts.append('exp')
                self.D.append(darg.as_poly(self.t, expand=False) * Poly(self.t, self.t, expand=False))
                if self.dummy:
                    i = Dummy('i')
                else:
                    i = Symbol('i')
                self.Tfuncs += [Lambda(i, exp(arg.subs(self.x, i)))]
                self.newf = self.newf.xreplace({exp(exparg): self.t ** p for exparg, p in others})
                new_extension = True
        if restart:
            return None
        return new_extension

    def _log_part(self, logs):
        """
        Try to build a logarithmic extension.

        Returns
        =======

        Returns True if there was a new extension and False if there was no new
        extension but it was able to rewrite the given logarithms in terms
        of the existing extension.  Unlike with exponential extensions, there
        is no way that a logarithm is not transcendental over and cannot be
        rewritten in terms of an already existing extension in a non-algebraic
        way, so this function does not ever return None or raise
        NotImplementedError.
        """
        from .prde import is_deriv_k
        new_extension = False
        logargs = [i.args[0] for i in logs]
        for arg in ordered(logargs):
            arga, argd = frac_in(arg, self.t)
            A = is_deriv_k(arga, argd, self)
            if A is not None:
                ans, u, const = A
                newterm = log(const) + u
                self.newf = self.newf.xreplace({log(arg): newterm})
                continue
            else:
                arga, argd = frac_in(arg, self.t)
                darga = argd * derivation(Poly(arga, self.t), self) - arga * derivation(Poly(argd, self.t), self)
                dargd = argd ** 2
                darg = darga.as_expr() / dargd.as_expr()
                self.t = next(self.ts)
                self.T.append(self.t)
                self.extargs.append(arg)
                self.exts.append('log')
                self.D.append(cancel(darg.as_expr() / arg).as_poly(self.t, expand=False))
                if self.dummy:
                    i = Dummy('i')
                else:
                    i = Symbol('i')
                self.Tfuncs += [Lambda(i, log(arg.subs(self.x, i)))]
                self.newf = self.newf.xreplace({log(arg): self.t})
                new_extension = True
        return new_extension

    @property
    def _important_attrs(self):
        """
        Returns some of the more important attributes of self.

        Explanation
        ===========

        Used for testing and debugging purposes.

        The attributes are (fa, fd, D, T, Tfuncs, backsubs,
        exts, extargs).
        """
        return (self.fa, self.fd, self.D, self.T, self.Tfuncs, self.backsubs, self.exts, self.extargs)

    def __repr__(self):
        r = [(attr, getattr(self, attr)) for attr in self.__slots__ if not isinstance(getattr(self, attr), GeneratorType)]
        return self.__class__.__name__ + '(dict(%r))' % r

    def __str__(self):
        return self.__class__.__name__ + '({fa=%s, fd=%s, D=%s})' % (self.fa, self.fd, self.D)

    def __eq__(self, other):
        for attr in self.__class__.__slots__:
            d1, d2 = (getattr(self, attr), getattr(other, attr))
            if not (isinstance(d1, GeneratorType) or d1 == d2):
                return False
        return True

    def reset(self):
        """
        Reset self to an initial state.  Used by __init__.
        """
        self.t = self.x
        self.T = [self.x]
        self.D = [Poly(1, self.x)]
        self.level = -1
        self.exts = [None]
        self.extargs = [None]
        if self.dummy:
            self.ts = numbered_symbols('t', cls=Dummy)
        else:
            self.ts = numbered_symbols('t')
        self.backsubs = []
        self.Tfuncs = []
        self.newf = self.f

    def indices(self, extension):
        """
        Parameters
        ==========

        extension : str
            Represents a valid extension type.

        Returns
        =======

        list: A list of indices of 'exts' where extension of
            type 'extension' is present.

        Examples
        ========

        >>> from sympy.integrals.risch import DifferentialExtension
        >>> from sympy import log, exp
        >>> from sympy.abc import x
        >>> DE = DifferentialExtension(log(x) + exp(x), x, handle_first='exp')
        >>> DE.indices('log')
        [2]
        >>> DE.indices('exp')
        [1]

        """
        return [i for i, ext in enumerate(self.exts) if ext == extension]

    def increment_level(self):
        """
        Increment the level of self.

        Explanation
        ===========

        This makes the working differential extension larger.  self.level is
        given relative to the end of the list (-1, -2, etc.), so we do not need
        do worry about it when building the extension.
        """
        if self.level >= -1:
            raise ValueError('The level of the differential extension cannot be incremented any further.')
        self.level += 1
        self.t = self.T[self.level]
        self.d = self.D[self.level]
        self.case = self.cases[self.level]
        return None

    def decrement_level(self):
        """
        Decrease the level of self.

        Explanation
        ===========

        This makes the working differential extension smaller.  self.level is
        given relative to the end of the list (-1, -2, etc.), so we do not need
        do worry about it when building the extension.
        """
        if self.level <= -len(self.T):
            raise ValueError('The level of the differential extension cannot be decremented any further.')
        self.level -= 1
        self.t = self.T[self.level]
        self.d = self.D[self.level]
        self.case = self.cases[self.level]
        return None