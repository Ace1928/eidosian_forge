import operator
import threading
import functools
import itertools
import contextlib
import collections
from ..autoray import (
from .draw import (
class LazyArray:
    """A lazy array representing a node in a computational graph.

    Parameters
    ----------
    backend : str
        The backend of the array were it to be computed. This can be ``None``
        but this may cause problems when propagating information about which
        functions to import to child nodes.
    fn : callable
        The function to call to compute the array, presumable imported from
        ``backend``. This can be ``None`` if the array already has data (e.g.
        is an input).
    args : tuple
        The positional arguments to pass to ``fn``, which might be
        ``LazyArray`` instances.
    kwargs : dict
        The keyword arguments to pass to ``fn``, which might be
        ``LazyArray`` instances.
    shape : tuple
        The shape of the array that ``fn(*args, **kwargs)`` will return, or
        the shape of the array that ``data`` has.
    deps : tuple, optional
        The ``LazyArray`` instances that ``fn(*args, **kwargs)`` depends on.
        If not specified, these will be automatically found from ``args`` and
        ``kwargs``, specifying them manually is slightly more efficient.

    """
    __slots__ = ('_backend', '_fn', '_args', '_kwargs', '_shape', '_data', '_deps', '_depth')

    def __init__(self, backend, fn, args, kwargs, shape, deps=None):
        self._backend = backend
        self._fn = fn
        self._args = args
        if kwargs is None:
            self._kwargs = _EMPTY_DICT
        else:
            self._kwargs = kwargs
        self._shape = shape
        self._data = None
        if deps is None:
            self._deps = (*find_lazy(self._args), *find_lazy(self._kwargs))
        else:
            self._deps = deps
        if self._deps:
            self._depth = max((d._depth for d in self._deps)) + 1
        else:
            self._depth = 0

    @classmethod
    def from_data(cls, data):
        """Create a new ``LazyArray`` directly from a concrete array."""
        obj = cls.__new__(cls)
        obj._backend = infer_backend(data)
        obj._fn = obj._args = obj._kwargs = None
        obj._shape = shape(data)
        obj._data = data
        obj._deps = ()
        obj._depth = 0
        return obj

    @classmethod
    def from_shape(cls, shape, backend='numpy'):
        """Create a new ``LazyArray`` with a given shape."""
        obj = cls.__new__(cls)
        obj._backend = backend
        obj._fn = obj._args = obj._kwargs = None
        obj._shape = tuple(map(int, shape))
        obj._data = PLACEHOLDER
        obj._deps = ()
        obj._depth = 0
        return obj

    def to(self, fn, args=None, kwargs=None, backend=None, shape=None, deps=None):
        """Create a new ``LazyArray``, by default propagating backend, shape,
        and deps from the the current LazyArray.
        """
        return LazyArray(fn=fn, args=args if args is not None else (self,), kwargs=kwargs, backend=backend if backend is not None else self._backend, shape=shape if shape is not None else self.shape, deps=deps if deps is not None else (self,))

    def _materialize(self):
        """Recursively compute all required args and kwargs for this node
        before computing itself and dereferencing dependencies. Note using this
        to materialize a large computation from scratch should be avoided due
        to the recursion limit, use ``x.compute()`` instead.
        """
        if self._data is None:
            args = (maybe_materialize(x) for x in self._args)
            kwargs = {k: maybe_materialize(v) for k, v in self._kwargs.items()}
            self._data = self._fn(*args, **kwargs)
            self._fn = self._args = self._kwargs = None
            self._deps = ()
        return self._data
    descend = descend
    ascend = ascend

    def compute(self):
        """Compute the value of this lazy array, clearing any references to the
        function, arguments and dependencies, and storing the result in the
        ``_data`` attribute as well as returning it.

        Unlike ``self._materialize()`` this avoids deep recursion.
        """
        for node in self.ascend():
            node._materialize()
        return self._data
    compute_constants = compute_constants

    def as_string(self, params):
        """Create a string which evaluates to the lazy array creation."""
        fn_name = f'{getattr(self._fn, '__name__', 'fn')}{id(self._fn)}'
        params.setdefault(fn_name, self._fn)
        str_call = ', '.join(itertools.chain((stringify(x, params) for x in self._args), (f'{k}: {stringify(v, params)}' for k, v in self._kwargs.items())))
        return f'x{id(self)} = {fn_name}({str_call})'
    get_source = get_source

    def get_function(self, variables, fold_constants=True):
        """Get a compiled function that computes ``fn(arrays)``, with ``fn``
        describing the computational graph of this ``LazyArray`` and ``arrays``
        corresponding to the downstream ``LazyArray`` nodes ``variables``.

        Parameters
        ----------
        variables : sequence of LazyArray
            Input nodes whose data can change between calls.
        fold_constants : bool, optional
            Compute all intermediates which do not depend on ``variables``
            prior to compilation.

        Returns
        -------
        fn : callable
            Function with signature ``fn(arrays)``.
        """
        return Function(inputs=variables, outputs=self, fold_constants=fold_constants)

    def show(self, filler=' ', max_lines=None, max_depth=None):
        """Show the computational graph as a nested directory structure."""
        if max_lines is None:
            max_lines = float('inf')
        if max_depth is None:
            max_depth = float('inf')
        bar = f'│{filler}'
        space = f'{filler}{filler}'
        junction = '├─'
        bend = '╰─'
        line = 0
        seen = {}
        queue = [(self, ())]
        while queue and line < max_lines:
            t, columns = queue.pop()
            prefix = ''
            if columns:
                prefix += ''.join((bar if not p else space for p in columns[:-1]))
                prefix += bend if columns[-1] else junction
            if t.fn_name not in (None, 'None'):
                item = f'{t.fn_name}{list(t.shape)}'
            else:
                item = f'←{list(t.shape)}'
            if t in seen:
                print(f'{line:>4} {prefix} ... ({item} from line {seen[t]})')
                line += 1
                continue
            print(f'{line:>4} {prefix}{item}')
            seen[t] = line
            line += 1
            if len(columns) < max_depth:
                deps = sorted(t.deps, key=get_depth, reverse=True)
                islasts = [True] + [False] * (len(deps) - 1)
                for islast, d in zip(islasts, deps):
                    queue.append((d, columns + (islast,)))

    def history_num_nodes(self):
        """Return the number of unique computational nodes in the history of
        this ``LazyArray``.
        """
        num_nodes = 0
        for _ in self.descend():
            num_nodes += 1
        return num_nodes

    def history_max_size(self):
        """Get the largest single tensor size appearing in this computation."""
        return max((node.size for node in self.descend()))

    def history_size_footprint(self, include_inputs=True):
        """Get the combined size of intermediates at each step of the
        computation. Note this assumes that intermediates are immediately
        garbage collected when they are no longer required.

        Parameters
        ----------
        include_inputs : bool, optional
            Whether to include the size of the inputs in the computation. If
            ``True`` It is assumed they can be garbage collected once used but
            are all present at the beginning of the computation.
        """
        delete_checked = set()
        sizes = []
        input_size = 0
        for node in reversed(tuple(self.ascend())):
            for c in node._deps:
                if c not in delete_checked:
                    if include_inputs or c._deps:
                        sizes.append(-c.size)
                    delete_checked.add(c)
            if node._data is None:
                sizes.append(+node.size)
            elif include_inputs:
                input_size += node.size
        sizes.append(input_size)
        sizes.reverse()
        return list(itertools.accumulate(sizes))

    def history_peak_size(self, include_inputs=True):
        """Get the peak combined intermediate size of this computation.

        Parameters
        ----------
        include_inputs : bool, optional
            Whether to include the size of the inputs in the computation. If
            ``True`` It is assumed they can be garbage collected once used but
            are all present at the beginning of the computation.
        """
        return max(self.history_size_footprint(include_inputs=include_inputs))

    def history_total_size(self):
        """The the total size of all unique arrays in the computational graph,
        possibly relevant e.g. for back-propagation algorithms.
        """
        return sum((node.size for node in self.descend()))

    def history_stats(self, fn):
        """Compute aggregate statistics about the computational graph.

        Parameters
        ----------
        fn : callable or str
            Function to apply to each node in the computational graph. If a
            string, one of 'count', 'sizein', 'sizeout' can be used to count
            the number of nodes, the total size of the inputs, or the total
            size of each output respectively.

        Returns
        -------
        stats : dict
            Dictionary mapping function names to the aggregate statistics.
        """
        if not callable(fn):
            if fn == 'count':

                def fn(node):
                    return 1
            elif fn == 'sizein':

                def fn(node):
                    return sum((child.size for child in node.deps))
            elif fn == 'sizeout':

                def fn(node):
                    return node.size
        stats = collections.defaultdict(int)
        for node in self.descend():
            node_cost = fn(node)
            if node_cost is not None:
                stats[node.fn_name] += fn(node)
        return dict(stats)

    def history_fn_frequencies(self):
        """Get a dictionary mapping function names to the number of times they
        are used in the computational graph.
        """
        return self.history_stats('count')

    def to_nx_digraph(self, variables=None):
        """Convert this ``LazyArray`` into a ``networkx.DiGraph``."""
        import networkx as nx
        if variables is None:
            variables = set()
        elif isinstance(variables, LazyArray):
            variables = {variables}
        else:
            variables = set(variables)
        G = nx.DiGraph()
        nodemap = {}
        for i, node in enumerate(self.ascend()):
            nodemap[node] = i
            variable = node in variables or any((child in variables for child in node.deps))
            if variable:
                variables.add(node)
            G.add_node(i, array=node, variable=variable)
            for x in node.deps:
                G.add_edge(nodemap[x], nodemap[node])
        return G
    plot = plot_circuit
    plot_graph = plot_graph
    plot_circuit = plot_circuit
    plot_history_size_footprint = plot_history_size_footprint
    plot_history_functions = plot_history_functions
    plot_history_functions_scatter = functools.partialmethod(plot_history_functions, kind='scatter')
    plot_history_functions_lines = functools.partialmethod(plot_history_functions, kind='lines')
    plot_history_functions_image = functools.partialmethod(plot_history_functions, kind='image')
    plot_history_stats = plot_history_stats
    plot_history_stats_counts = functools.partialmethod(plot_history_stats, fn='count')
    plot_history_stats_sizein = functools.partialmethod(plot_history_stats, fn='sizein')

    @property
    def fn(self):
        """The function to use to compute this array."""
        return self._fn

    @property
    def fn_name(self):
        """The name of the function to use to compute this array."""
        return getattr(self._fn, '__name__', 'None')

    @property
    def args(self):
        """The positional arguments to the function to use to compute this
        array.
        """
        return self._args

    @property
    def kwargs(self):
        """The keyword arguments to the function to use to compute this
        array.
        """
        return self._kwargs

    @property
    def shape(self):
        return self._shape

    def __len__(self):
        return self.shape[0]

    def __iter__(self):
        import warnings
        warnings.warn('Iterating over LazyArray to get the computational graph nodes is deprecated - use `LazyArray.descend()` instead. Eventually `iter(lz)` will iterate over first axis slices.')
        return self.descend()

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def size(self):
        return functools.reduce(operator.mul, self.shape, 1)

    @property
    def backend(self):
        return self._backend

    @property
    def deps(self):
        """A tuple of the dependencies, other LazyArray instances, of this
        array.
        """
        return self._deps

    @property
    def depth(self):
        """The maximum distance to any input array in the computational graph.
        """
        return self._depth

    def __getitem__(self, key):
        return getitem(self, key)
    __array_ufunc__ = None

    def __mul__(self, other):
        return multiply(self, other)

    def __rmul__(self, other):
        return multiply(self, other)

    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(self, other)

    def __sub__(self, other):
        return sub(self, other)

    def __rsub__(self, other):
        return sub(other, self)

    def __floordiv__(self, other):
        return floordivide(self, other)

    def __rfloordiv__(self, other):
        return floordivide(other, self)

    def __truediv__(self, other):
        return truedivide(self, other)

    def __rtruediv__(self, other):
        return truedivide(other, self)

    def __pow__(self, other):
        return pow_(self, other)

    def __rpow__(self, other):
        return pow_(other, self)

    def __matmul__(self, other):
        return matmul(self, other)

    def __rmatmul__(self, other):
        return matmul(other, self)

    def __abs__(self):
        return abs_(self)

    def __neg__(self):
        return self.to(operator.neg)

    def __ne__(self, other):
        return ne(self, other)

    def __gt__(self, other):
        return gt(self, other)

    def __lt__(self, other):
        return lt(self, other)

    def __ge__(self, other):
        return ge(self, other)

    def __le__(self, other):
        return le(self, other)

    @property
    def T(self):
        return transpose(self)

    @property
    def H(self):
        return conj(transpose(self))

    def reshape(self, shape):
        return reshape(self, shape)

    def astype(self, dtype_name):
        return lazy_astype(self, dtype_name)

    @property
    def real(self):
        return real(self)

    @property
    def imag(self):
        return imag(self)

    def __repr__(self):
        return f"<{self.__class__.__name__}(fn={self.fn_name}, shape={self.shape}, backend='{self.backend}')>"