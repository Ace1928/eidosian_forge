from collections import defaultdict
from .cache import cacheit
from sympy.multipledispatch.dispatcher import (Dispatcher,
class KindDispatcher:
    """
    Dispatcher to select a kind from multiple kinds by binary dispatching.

    .. notes::
       This approach is experimental, and can be replaced or deleted in
       the future.

    Explanation
    ===========

    SymPy object's :obj:`sympy.core.kind.Kind()` vaguely represents the
    algebraic structure where the object belongs to. Therefore, with
    given operation, we can always find a dominating kind among the
    different kinds. This class selects the kind by recursive binary
    dispatching. If the result cannot be determined, ``UndefinedKind``
    is returned.

    Examples
    ========

    Multiplication between numbers return number.

    >>> from sympy import NumberKind, Mul
    >>> Mul._kind_dispatcher(NumberKind, NumberKind)
    NumberKind

    Multiplication between number and unknown-kind object returns unknown kind.

    >>> from sympy import UndefinedKind
    >>> Mul._kind_dispatcher(NumberKind, UndefinedKind)
    UndefinedKind

    Any number and order of kinds is allowed.

    >>> Mul._kind_dispatcher(UndefinedKind, NumberKind)
    UndefinedKind
    >>> Mul._kind_dispatcher(NumberKind, UndefinedKind, NumberKind)
    UndefinedKind

    Since matrix forms a vector space over scalar field, multiplication
    between matrix with numeric element and number returns matrix with
    numeric element.

    >>> from sympy.matrices import MatrixKind
    >>> Mul._kind_dispatcher(MatrixKind(NumberKind), NumberKind)
    MatrixKind(NumberKind)

    If a matrix with number element and another matrix with unknown-kind
    element are multiplied, we know that the result is matrix but the
    kind of its elements is unknown.

    >>> Mul._kind_dispatcher(MatrixKind(NumberKind), MatrixKind(UndefinedKind))
    MatrixKind(UndefinedKind)

    Parameters
    ==========

    name : str

    commutative : bool, optional
        If True, binary dispatch will be automatically registered in
        reversed order as well.

    doc : str, optional

    """

    def __init__(self, name, commutative=False, doc=None):
        self.name = name
        self.doc = doc
        self.commutative = commutative
        self._dispatcher = Dispatcher(name)

    def __repr__(self):
        return '<dispatched %s>' % self.name

    def register(self, *types, **kwargs):
        """
        Register the binary dispatcher for two kind classes.

        If *self.commutative* is ``True``, signature in reversed order is
        automatically registered as well.
        """
        on_ambiguity = kwargs.pop('on_ambiguity', None)
        if not on_ambiguity:
            if self.commutative:
                on_ambiguity = ambiguity_register_error_ignore_dup
            else:
                on_ambiguity = ambiguity_warn
        kwargs.update(on_ambiguity=on_ambiguity)
        if not len(types) == 2:
            raise RuntimeError('Only binary dispatch is supported, but got %s types: <%s>.' % (len(types), str_signature(types)))

        def _(func):
            self._dispatcher.add(types, func, **kwargs)
            if self.commutative:
                self._dispatcher.add(tuple(reversed(types)), func, **kwargs)
        return _

    def __call__(self, *args, **kwargs):
        if self.commutative:
            kinds = frozenset(args)
        else:
            kinds = []
            prev = None
            for a in args:
                if prev is not a:
                    kinds.append(a)
                    prev = a
        return self.dispatch_kinds(kinds, **kwargs)

    @cacheit
    def dispatch_kinds(self, kinds, **kwargs):
        if len(kinds) == 1:
            result, = kinds
            if not isinstance(result, Kind):
                raise RuntimeError('%s is not a kind.' % result)
            return result
        for i, kind in enumerate(kinds):
            if not isinstance(kind, Kind):
                raise RuntimeError('%s is not a kind.' % kind)
            if i == 0:
                result = kind
            else:
                prev_kind = result
                t1, t2 = (type(prev_kind), type(kind))
                k1, k2 = (prev_kind, kind)
                func = self._dispatcher.dispatch(t1, t2)
                if func is None and self.commutative:
                    func = self._dispatcher.dispatch(t2, t1)
                    k1, k2 = (k2, k1)
                if func is None:
                    result = UndefinedKind
                else:
                    result = func(k1, k2)
                if not isinstance(result, Kind):
                    raise RuntimeError('Dispatcher for {!r} and {!r} must return a Kind, but got {!r}'.format(prev_kind, kind, result))
        return result

    @property
    def __doc__(self):
        docs = ['Kind dispatcher : %s' % self.name, 'Note that support for this is experimental. See the docs for :class:`KindDispatcher` for details']
        if self.doc:
            docs.append(self.doc)
        s = 'Registered kind classes\n'
        s += '=' * len(s)
        docs.append(s)
        amb_sigs = []
        typ_sigs = defaultdict(list)
        for sigs in self._dispatcher.ordering[::-1]:
            key = self._dispatcher.funcs[sigs]
            typ_sigs[key].append(sigs)
        for func, sigs in typ_sigs.items():
            sigs_str = ', '.join(('<%s>' % str_signature(sig) for sig in sigs))
            if isinstance(func, RaiseNotImplementedError):
                amb_sigs.append(sigs_str)
                continue
            s = 'Inputs: %s\n' % sigs_str
            s += '-' * len(s) + '\n'
            if func.__doc__:
                s += func.__doc__.strip()
            else:
                s += func.__name__
            docs.append(s)
        if amb_sigs:
            s = 'Ambiguous kind classes\n'
            s += '=' * len(s)
            docs.append(s)
            s = '\n'.join(amb_sigs)
            docs.append(s)
        return '\n\n'.join(docs)