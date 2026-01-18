import inspect
import itertools
import os
import warnings
from functools import partial
from importlib.metadata import entry_points
import networkx as nx
from .decorators import argmap
from .configs import Config, config
class _dispatchable:
    """Allow any of the following decorator forms:
    - @_dispatchable
    - @_dispatchable()
    - @_dispatchable(name="override_name")
    - @_dispatchable(graphs="graph")
    - @_dispatchable(edge_attrs="weight")
    - @_dispatchable(graphs={"G": 0, "H": 1}, edge_attrs={"weight": "default"})

    These class attributes are currently used to allow backends to run networkx tests.
    For example: `PYTHONPATH=. pytest --backend graphblas --fallback-to-nx`
    Future work: add configuration to control these.
    """
    _is_testing = False
    _fallback_to_nx = os.environ.get('NETWORKX_FALLBACK_TO_NX', 'true').strip().lower() == 'true'

    def __new__(cls, func=None, *, name=None, graphs='G', edge_attrs=None, node_attrs=None, preserve_edge_attrs=False, preserve_node_attrs=False, preserve_graph_attrs=False, preserve_all_attrs=False, mutates_input=False, returns_graph=False):
        """A decorator that makes certain input graph types dispatch to ``func``'s
        backend implementation.

        Usage can be any of the following decorator forms:
        - @_dispatchable
        - @_dispatchable()
        - @_dispatchable(name="override_name")
        - @_dispatchable(graphs="graph_var_name")
        - @_dispatchable(edge_attrs="weight")
        - @_dispatchable(graphs={"G": 0, "H": 1}, edge_attrs={"weight": "default"})
        with 0 and 1 giving the position in the signature function for graph objects.
        When edge_attrs is a dict, keys are keyword names and values are defaults.

        The class attributes are used to allow backends to run networkx tests.
        For example: `PYTHONPATH=. pytest --backend graphblas --fallback-to-nx`
        Future work: add configuration to control these.

        Parameters
        ----------
        func : callable, optional
            The function to be decorated. If ``func`` is not provided, returns a
            partial object that can be used to decorate a function later. If ``func``
            is provided, returns a new callable object that dispatches to a backend
            algorithm based on input graph types.

        name : str, optional
            The name of the algorithm to use for dispatching. If not provided,
            the name of ``func`` will be used. ``name`` is useful to avoid name
            conflicts, as all dispatched algorithms live in a single namespace.
            For example, ``tournament.is_strongly_connected`` had a name conflict
            with the standard ``nx.is_strongly_connected``, so we used
            ``@_dispatchable(name="tournament_is_strongly_connected")``.

        graphs : str or dict or None, default "G"
            If a string, the parameter name of the graph, which must be the first
            argument of the wrapped function. If more than one graph is required
            for the algorithm (or if the graph is not the first argument), provide
            a dict of parameter name to argument position for each graph argument.
            For example, ``@_dispatchable(graphs={"G": 0, "auxiliary?": 4})``
            indicates the 0th parameter ``G`` of the function is a required graph,
            and the 4th parameter ``auxiliary`` is an optional graph.
            To indicate an argument is a list of graphs, do e.g. ``"[graphs]"``.
            Use ``graphs=None`` if *no* arguments are NetworkX graphs such as for
            graph generators, readers, and conversion functions.

        edge_attrs : str or dict, optional
            ``edge_attrs`` holds information about edge attribute arguments
            and default values for those edge attributes.
            If a string, ``edge_attrs`` holds the function argument name that
            indicates a single edge attribute to include in the converted graph.
            The default value for this attribute is 1. To indicate that an argument
            is a list of attributes (all with default value 1), use e.g. ``"[attrs]"``.
            If a dict, ``edge_attrs`` holds a dict keyed by argument names, with
            values that are either the default value or, if a string, the argument
            name that indicates the default value.

        node_attrs : str or dict, optional
            Like ``edge_attrs``, but for node attributes.

        preserve_edge_attrs : bool or str or dict, optional
            For bool, whether to preserve all edge attributes.
            For str, the parameter name that may indicate (with ``True`` or a
            callable argument) whether all edge attributes should be preserved
            when converting.
            For dict of ``{graph_name: {attr: default}}``, indicate pre-determined
            edge attributes (and defaults) to preserve for input graphs.

        preserve_node_attrs : bool or str or dict, optional
            Like ``preserve_edge_attrs``, but for node attributes.

        preserve_graph_attrs : bool or set
            For bool, whether to preserve all graph attributes.
            For set, which input graph arguments to preserve graph attributes.

        preserve_all_attrs : bool
            Whether to preserve all edge, node and graph attributes.
            This overrides all the other preserve_*_attrs.

        mutates_input : bool or dict, default False
            For bool, whether the functions mutates an input graph argument.
            For dict of ``{arg_name: arg_pos}``, arguments that indicates whether an
            input graph will be mutated, and ``arg_name`` may begin with ``"not "``
            to negate the logic (for example, this is used by ``copy=`` arguments).
            By default, dispatching doesn't convert input graphs to a different
            backend for functions that mutate input graphs.

        returns_graph : bool, default False
            Whether the function can return or yield a graph object. By default,
            dispatching doesn't convert input graphs to a different backend for
            functions that return graphs.
        """
        if func is None:
            return partial(_dispatchable, name=name, graphs=graphs, edge_attrs=edge_attrs, node_attrs=node_attrs, preserve_edge_attrs=preserve_edge_attrs, preserve_node_attrs=preserve_node_attrs, preserve_graph_attrs=preserve_graph_attrs, preserve_all_attrs=preserve_all_attrs, mutates_input=mutates_input, returns_graph=returns_graph)
        if isinstance(func, str):
            raise TypeError("'name' and 'graphs' must be passed by keyword") from None
        if name is None:
            name = func.__name__
        self = object.__new__(cls)
        self.__name__ = func.__name__
        self.__defaults__ = func.__defaults__
        if func.__kwdefaults__:
            self.__kwdefaults__ = {**func.__kwdefaults__, 'backend': None}
        else:
            self.__kwdefaults__ = {'backend': None}
        self.__module__ = func.__module__
        self.__qualname__ = func.__qualname__
        self.__dict__.update(func.__dict__)
        self.__wrapped__ = func
        self._orig_doc = func.__doc__
        self._cached_doc = None
        self.orig_func = func
        self.name = name
        self.edge_attrs = edge_attrs
        self.node_attrs = node_attrs
        self.preserve_edge_attrs = preserve_edge_attrs or preserve_all_attrs
        self.preserve_node_attrs = preserve_node_attrs or preserve_all_attrs
        self.preserve_graph_attrs = preserve_graph_attrs or preserve_all_attrs
        self.mutates_input = mutates_input
        self._returns_graph = returns_graph
        if edge_attrs is not None and (not isinstance(edge_attrs, str | dict)):
            raise TypeError(f'Bad type for edge_attrs: {type(edge_attrs)}. Expected str or dict.') from None
        if node_attrs is not None and (not isinstance(node_attrs, str | dict)):
            raise TypeError(f'Bad type for node_attrs: {type(node_attrs)}. Expected str or dict.') from None
        if not isinstance(self.preserve_edge_attrs, bool | str | dict):
            raise TypeError(f'Bad type for preserve_edge_attrs: {type(self.preserve_edge_attrs)}. Expected bool, str, or dict.') from None
        if not isinstance(self.preserve_node_attrs, bool | str | dict):
            raise TypeError(f'Bad type for preserve_node_attrs: {type(self.preserve_node_attrs)}. Expected bool, str, or dict.') from None
        if not isinstance(self.preserve_graph_attrs, bool | set):
            raise TypeError(f'Bad type for preserve_graph_attrs: {type(self.preserve_graph_attrs)}. Expected bool or set.') from None
        if not isinstance(self.mutates_input, bool | dict):
            raise TypeError(f'Bad type for mutates_input: {type(self.mutates_input)}. Expected bool or dict.') from None
        if not isinstance(self._returns_graph, bool):
            raise TypeError(f'Bad type for returns_graph: {type(self._returns_graph)}. Expected bool.') from None
        if isinstance(graphs, str):
            graphs = {graphs: 0}
        elif graphs is None:
            pass
        elif not isinstance(graphs, dict):
            raise TypeError(f'Bad type for graphs: {type(graphs)}. Expected str or dict.') from None
        elif len(graphs) == 0:
            raise KeyError("'graphs' must contain at least one variable name") from None
        self.optional_graphs = set()
        self.list_graphs = set()
        if graphs is None:
            self.graphs = {}
        else:
            self.graphs = {self.optional_graphs.add((val := k[:-1])) or val if (last := k[-1]) == '?' else self.list_graphs.add((val := k[1:-1])) or val if last == ']' else k: v for k, v in graphs.items()}
        self._sig = None
        self.backends = {backend for backend, info in backend_info.items() if 'functions' in info and name in info['functions']}
        if name in _registered_algorithms:
            raise KeyError(f'Algorithm already exists in dispatch registry: {name}') from None
        self = argmap(_do_nothing)(self)
        _registered_algorithms[name] = self
        return self

    @property
    def __doc__(self):
        """If the cached documentation exists, it is returned.
        Otherwise, the documentation is generated using _make_doc() method,
        cached, and then returned."""
        if (rv := self._cached_doc) is not None:
            return rv
        rv = self._cached_doc = self._make_doc()
        return rv

    @__doc__.setter
    def __doc__(self, val):
        """Sets the original documentation to the given value and resets the
        cached documentation."""
        self._orig_doc = val
        self._cached_doc = None

    @property
    def __signature__(self):
        """Return the signature of the original function, with the addition of
        the `backend` and `backend_kwargs` parameters."""
        if self._sig is None:
            sig = inspect.signature(self.orig_func)
            if not any((p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())):
                sig = sig.replace(parameters=[*sig.parameters.values(), inspect.Parameter('backend', inspect.Parameter.KEYWORD_ONLY, default=None), inspect.Parameter('backend_kwargs', inspect.Parameter.VAR_KEYWORD)])
            else:
                *parameters, var_keyword = sig.parameters.values()
                sig = sig.replace(parameters=[*parameters, inspect.Parameter('backend', inspect.Parameter.KEYWORD_ONLY, default=None), var_keyword])
            self._sig = sig
        return self._sig

    def __call__(self, /, *args, backend=None, **kwargs):
        """Returns the result of the original function, or the backend function if
        the backend is specified and that backend implements `func`."""
        if not backends:
            return self.orig_func(*args, **kwargs)
        backend_name = backend
        if backend_name is not None and backend_name not in backends:
            raise ImportError(f'Unable to load backend: {backend_name}')
        graphs_resolved = {}
        for gname, pos in self.graphs.items():
            if pos < len(args):
                if gname in kwargs:
                    raise TypeError(f'{self.name}() got multiple values for {gname!r}')
                val = args[pos]
            elif gname in kwargs:
                val = kwargs[gname]
            elif gname not in self.optional_graphs:
                raise TypeError(f'{self.name}() missing required graph argument: {gname}')
            else:
                continue
            if val is None:
                if gname not in self.optional_graphs:
                    raise TypeError(f'{self.name}() required graph argument {gname!r} is None; must be a graph')
            else:
                graphs_resolved[gname] = val
        if self.list_graphs:
            args = list(args)
            for gname in self.list_graphs & graphs_resolved.keys():
                val = list(graphs_resolved[gname])
                graphs_resolved[gname] = val
                if gname in kwargs:
                    kwargs[gname] = val
                else:
                    args[self.graphs[gname]] = val
            has_backends = any((hasattr(g, '__networkx_backend__') if gname not in self.list_graphs else any((hasattr(g2, '__networkx_backend__') for g2 in g)) for gname, g in graphs_resolved.items()))
            if has_backends:
                graph_backend_names = {getattr(g, '__networkx_backend__', 'networkx') for gname, g in graphs_resolved.items() if gname not in self.list_graphs}
                for gname in self.list_graphs & graphs_resolved.keys():
                    graph_backend_names.update((getattr(g, '__networkx_backend__', 'networkx') for g in graphs_resolved[gname]))
        else:
            has_backends = any((hasattr(g, '__networkx_backend__') for g in graphs_resolved.values()))
            if has_backends:
                graph_backend_names = {getattr(g, '__networkx_backend__', 'networkx') for g in graphs_resolved.values()}
        backend_priority = config.backend_priority
        if self._is_testing and backend_priority and (backend_name is None):
            return self._convert_and_call_for_tests(backend_priority[0], args, kwargs, fallback_to_nx=self._fallback_to_nx)
        if has_backends:
            backend_names = graph_backend_names - {'networkx'}
            if len(backend_names) != 1:
                raise TypeError(f'{self.name}() graphs must all be from the same backend, found {backend_names}')
            [graph_backend_name] = backend_names
            if backend_name is not None and backend_name != graph_backend_name:
                raise TypeError(f'{self.name}() is unable to convert graph from backend {graph_backend_name!r} to the specified backend {backend_name!r}.')
            if graph_backend_name not in backends:
                raise ImportError(f'Unable to load backend: {graph_backend_name}')
            if 'networkx' in graph_backend_names and graph_backend_name not in backend_priority:
                raise TypeError(f'Unable to convert inputs and run {self.name}. {self.name}() has networkx and {graph_backend_name} graphs, but NetworkX is not configured to automatically convert graphs from networkx to {graph_backend_name}.')
            backend = _load_backend(graph_backend_name)
            if hasattr(backend, self.name):
                if 'networkx' in graph_backend_names:
                    return self._convert_and_call(graph_backend_name, args, kwargs, fallback_to_nx=self._fallback_to_nx)
                return getattr(backend, self.name)(*args, **kwargs)
            raise nx.NetworkXNotImplemented(f"'{self.name}' not implemented by {graph_backend_name}")
        if backend_name is not None:
            return self._convert_and_call(backend_name, args, kwargs, fallback_to_nx=False)
        if not self._returns_graph and (not self.mutates_input or (isinstance(self.mutates_input, dict) and any((not (args[arg_pos] if len(args) > arg_pos else kwargs.get(arg_name[4:], True)) if arg_name.startswith('not ') else (args[arg_pos] if len(args) > arg_pos else kwargs.get(arg_name)) is not None for arg_name, arg_pos in self.mutates_input.items())))):
            for backend_name in backend_priority:
                if self._should_backend_run(backend_name, *args, **kwargs):
                    return self._convert_and_call(backend_name, args, kwargs, fallback_to_nx=self._fallback_to_nx)
        return self.orig_func(*args, **kwargs)

    def _can_backend_run(self, backend_name, /, *args, **kwargs):
        """Can the specified backend run this algorithm with these arguments?"""
        backend = _load_backend(backend_name)
        return hasattr(backend, self.name) and (can_run := backend.can_run(self.name, args, kwargs)) and (not isinstance(can_run, str))

    def _should_backend_run(self, backend_name, /, *args, **kwargs):
        """Can/should the specified backend run this algorithm with these arguments?"""
        backend = _load_backend(backend_name)
        return hasattr(backend, self.name) and (can_run := backend.can_run(self.name, args, kwargs)) and (not isinstance(can_run, str)) and (should_run := backend.should_run(self.name, args, kwargs)) and (not isinstance(should_run, str))

    def _convert_arguments(self, backend_name, args, kwargs, *, use_cache):
        """Convert graph arguments to the specified backend.

        Returns
        -------
        args tuple and kwargs dict
        """
        bound = self.__signature__.bind(*args, **kwargs)
        bound.apply_defaults()
        if not self.graphs:
            bound_kwargs = bound.kwargs
            del bound_kwargs['backend']
            return (bound.args, bound_kwargs)
        preserve_edge_attrs = self.preserve_edge_attrs
        edge_attrs = self.edge_attrs
        if preserve_edge_attrs is False:
            pass
        elif preserve_edge_attrs is True:
            edge_attrs = None
        elif isinstance(preserve_edge_attrs, str):
            if bound.arguments[preserve_edge_attrs] is True or callable(bound.arguments[preserve_edge_attrs]):
                preserve_edge_attrs = True
                edge_attrs = None
            elif bound.arguments[preserve_edge_attrs] is False and (isinstance(edge_attrs, str) and edge_attrs == preserve_edge_attrs or (isinstance(edge_attrs, dict) and preserve_edge_attrs in edge_attrs)):
                preserve_edge_attrs = False
                edge_attrs = None
            else:
                preserve_edge_attrs = False
        if edge_attrs is None:
            pass
        elif isinstance(edge_attrs, str):
            if edge_attrs[0] == '[':
                edge_attrs = {edge_attr: 1 for edge_attr in bound.arguments[edge_attrs[1:-1]]}
            elif callable(bound.arguments[edge_attrs]):
                preserve_edge_attrs = True
                edge_attrs = None
            elif bound.arguments[edge_attrs] is not None:
                edge_attrs = {bound.arguments[edge_attrs]: 1}
            elif self.name == 'to_numpy_array' and hasattr(bound.arguments['dtype'], 'names'):
                edge_attrs = {edge_attr: 1 for edge_attr in bound.arguments['dtype'].names}
            else:
                edge_attrs = None
        else:
            edge_attrs = {edge_attr: bound.arguments.get(val, 1) if isinstance(val, str) else val for key, val in edge_attrs.items() if (edge_attr := bound.arguments[key]) is not None}
        preserve_node_attrs = self.preserve_node_attrs
        node_attrs = self.node_attrs
        if preserve_node_attrs is False:
            pass
        elif preserve_node_attrs is True:
            node_attrs = None
        elif isinstance(preserve_node_attrs, str):
            if bound.arguments[preserve_node_attrs] is True or callable(bound.arguments[preserve_node_attrs]):
                preserve_node_attrs = True
                node_attrs = None
            elif bound.arguments[preserve_node_attrs] is False and (isinstance(node_attrs, str) and node_attrs == preserve_node_attrs or (isinstance(node_attrs, dict) and preserve_node_attrs in node_attrs)):
                preserve_node_attrs = False
                node_attrs = None
            else:
                preserve_node_attrs = False
        if node_attrs is None:
            pass
        elif isinstance(node_attrs, str):
            if node_attrs[0] == '[':
                node_attrs = {node_attr: None for node_attr in bound.arguments[node_attrs[1:-1]]}
            elif callable(bound.arguments[node_attrs]):
                preserve_node_attrs = True
                node_attrs = None
            elif bound.arguments[node_attrs] is not None:
                node_attrs = {bound.arguments[node_attrs]: None}
            else:
                node_attrs = None
        else:
            node_attrs = {node_attr: bound.arguments.get(val) if isinstance(val, str) else val for key, val in node_attrs.items() if (node_attr := bound.arguments[key]) is not None}
        preserve_graph_attrs = self.preserve_graph_attrs
        for gname in self.graphs:
            if gname in self.list_graphs:
                bound.arguments[gname] = [self._convert_graph(backend_name, g, edge_attrs=edge_attrs, node_attrs=node_attrs, preserve_edge_attrs=preserve_edge_attrs, preserve_node_attrs=preserve_node_attrs, preserve_graph_attrs=preserve_graph_attrs, graph_name=gname, use_cache=use_cache) if getattr(g, '__networkx_backend__', 'networkx') == 'networkx' else g for g in bound.arguments[gname]]
            else:
                graph = bound.arguments[gname]
                if graph is None:
                    if gname in self.optional_graphs:
                        continue
                    raise TypeError(f'Missing required graph argument `{gname}` in {self.name} function')
                if isinstance(preserve_edge_attrs, dict):
                    preserve_edges = False
                    edges = preserve_edge_attrs.get(gname, edge_attrs)
                else:
                    preserve_edges = preserve_edge_attrs
                    edges = edge_attrs
                if isinstance(preserve_node_attrs, dict):
                    preserve_nodes = False
                    nodes = preserve_node_attrs.get(gname, node_attrs)
                else:
                    preserve_nodes = preserve_node_attrs
                    nodes = node_attrs
                if isinstance(preserve_graph_attrs, set):
                    preserve_graph = gname in preserve_graph_attrs
                else:
                    preserve_graph = preserve_graph_attrs
                if getattr(graph, '__networkx_backend__', 'networkx') == 'networkx':
                    bound.arguments[gname] = self._convert_graph(backend_name, graph, edge_attrs=edges, node_attrs=nodes, preserve_edge_attrs=preserve_edges, preserve_node_attrs=preserve_nodes, preserve_graph_attrs=preserve_graph, graph_name=gname, use_cache=use_cache)
        bound_kwargs = bound.kwargs
        del bound_kwargs['backend']
        return (bound.args, bound_kwargs)

    def _convert_graph(self, backend_name, graph, *, edge_attrs, node_attrs, preserve_edge_attrs, preserve_node_attrs, preserve_graph_attrs, graph_name, use_cache):
        if use_cache and (nx_cache := getattr(graph, '__networkx_cache__', None)) is not None:
            cache = nx_cache.setdefault('backends', {}).setdefault(backend_name, {})
            key = edge_key, node_key, graph_key = (frozenset(edge_attrs.items()) if edge_attrs is not None else preserve_edge_attrs, frozenset(node_attrs.items()) if node_attrs is not None else preserve_node_attrs, preserve_graph_attrs)
            if cache:
                warning_message = f'Using cached graph for {backend_name!r} backend in call to {self.name}.\n\nFor the cache to be consistent (i.e., correct), the input graph must not have been manually mutated since the cached graph was created. Examples of manually mutating the graph data structures resulting in an inconsistent cache include:\n\n    >>> G[u][v][key] = val\n\nand\n\n    >>> for u, v, d in G.edges(data=True):\n    ...     d[key] = val\n\nUsing methods such as `G.add_edge(u, v, weight=val)` will correctly clear the cache to keep it consistent. You may also use `G.__networkx_cache__.clear()` to manually clear the cache, or set `G.__networkx_cache__` to None to disable caching for G. Enable or disable caching via `nx.config.cache_converted_graphs` config.'
                for compat_key in itertools.product((edge_key, True) if edge_key is not True else (True,), (node_key, True) if node_key is not True else (True,), (graph_key, True) if graph_key is not True else (True,)):
                    if (rv := cache.get(compat_key)) is not None:
                        warnings.warn(warning_message)
                        return rv
                if edge_key is not True and node_key is not True:
                    for (ekey, nkey, gkey), val in list(cache.items()):
                        if edge_key is False or ekey is True:
                            pass
                        elif edge_key is True or ekey is False or (not edge_key.issubset(ekey)):
                            continue
                        if node_key is False or nkey is True:
                            pass
                        elif node_key is True or nkey is False or (not node_key.issubset(nkey)):
                            continue
                        if graph_key and (not gkey):
                            continue
                        warnings.warn(warning_message)
                        return val
        backend = _load_backend(backend_name)
        rv = backend.convert_from_nx(graph, edge_attrs=edge_attrs, node_attrs=node_attrs, preserve_edge_attrs=preserve_edge_attrs, preserve_node_attrs=preserve_node_attrs, preserve_graph_attrs=preserve_graph_attrs, name=self.name, graph_name=graph_name)
        if use_cache and nx_cache is not None:
            cache[key] = rv
            for cur_key in list(cache):
                if cur_key == key:
                    continue
                ekey, nkey, gkey = cur_key
                if ekey is False or edge_key is True:
                    pass
                elif ekey is True or edge_key is False or (not ekey.issubset(edge_key)):
                    continue
                if nkey is False or node_key is True:
                    pass
                elif nkey is True or node_key is False or (not nkey.issubset(node_key)):
                    continue
                if gkey and (not graph_key):
                    continue
                cache.pop(cur_key, None)
        return rv

    def _convert_and_call(self, backend_name, args, kwargs, *, fallback_to_nx=False):
        """Call this dispatchable function with a backend, converting graphs if necessary."""
        backend = _load_backend(backend_name)
        if not self._can_backend_run(backend_name, *args, **kwargs):
            if fallback_to_nx:
                return self.orig_func(*args, **kwargs)
            msg = f"'{self.name}' not implemented by {backend_name}"
            if hasattr(backend, self.name):
                msg += ' with the given arguments'
            raise RuntimeError(msg)
        try:
            converted_args, converted_kwargs = self._convert_arguments(backend_name, args, kwargs, use_cache=config.cache_converted_graphs)
            result = getattr(backend, self.name)(*converted_args, **converted_kwargs)
        except (NotImplementedError, nx.NetworkXNotImplemented) as exc:
            if fallback_to_nx:
                return self.orig_func(*args, **kwargs)
            raise
        return result

    def _convert_and_call_for_tests(self, backend_name, args, kwargs, *, fallback_to_nx=False):
        """Call this dispatchable function with a backend; for use with testing."""
        backend = _load_backend(backend_name)
        if not self._can_backend_run(backend_name, *args, **kwargs):
            if fallback_to_nx or not self.graphs:
                return self.orig_func(*args, **kwargs)
            import pytest
            msg = f"'{self.name}' not implemented by {backend_name}"
            if hasattr(backend, self.name):
                msg += ' with the given arguments'
            pytest.xfail(msg)
        from collections.abc import Iterable, Iterator, Mapping
        from copy import copy
        from io import BufferedReader, BytesIO, StringIO, TextIOWrapper
        from itertools import tee
        from random import Random
        import numpy as np
        from numpy.random import Generator, RandomState
        from scipy.sparse import sparray
        if not args:
            args1 = args2 = args
        else:
            args1, args2 = zip(*((arg, copy(arg)) if isinstance(arg, BytesIO | StringIO | Random | Generator | RandomState) else tee(arg) if isinstance(arg, Iterator) and (not isinstance(arg, BufferedReader | TextIOWrapper)) else (arg, arg) for arg in args))
        if not kwargs:
            kwargs1 = kwargs2 = kwargs
        else:
            kwargs1, kwargs2 = zip(*(((k, v), (k, copy(v))) if isinstance(v, BytesIO | StringIO | Random | Generator | RandomState) else ((k, (teed := tee(v))[0]), (k, teed[1])) if isinstance(v, Iterator) and (not isinstance(v, BufferedReader | TextIOWrapper)) else ((k, v), (k, v)) for k, v in kwargs.items()))
            kwargs1 = dict(kwargs1)
            kwargs2 = dict(kwargs2)
        try:
            converted_args, converted_kwargs = self._convert_arguments(backend_name, args1, kwargs1, use_cache=False)
            result = getattr(backend, self.name)(*converted_args, **converted_kwargs)
        except (NotImplementedError, nx.NetworkXNotImplemented) as exc:
            if fallback_to_nx:
                return self.orig_func(*args2, **kwargs2)
            import pytest
            pytest.xfail(exc.args[0] if exc.args else f'{self.name} raised {type(exc).__name__}')
        if self._returns_graph != (isinstance(result, nx.Graph) or hasattr(result, '__networkx_backend__') or (isinstance(result, tuple | list) and any((isinstance(x, nx.Graph) or hasattr(x, '__networkx_backend__') for x in result)))) and (not (self.name in {'check_planarity', 'check_planarity_recursive'} and any((x is None for x in result)))) and (not (self.name in {'held_karp_ascent'} and any((isinstance(x, dict) for x in result)))) and (self.name not in {'all_triads', 'general_k_edge_subgraphs', 'nonisomorphic_trees'}):
            raise RuntimeError(f'`returns_graph` is incorrect for {self.name}')

        def check_result(val, depth=0):
            if isinstance(val, np.number):
                raise RuntimeError(f'{self.name} returned a numpy scalar {val} ({type(val)}, depth={depth})')
            if isinstance(val, np.ndarray | sparray):
                return
            if isinstance(val, nx.Graph):
                check_result(val._node, depth=depth + 1)
                check_result(val._adj, depth=depth + 1)
                return
            if isinstance(val, Iterator):
                raise NotImplementedError
            if isinstance(val, Iterable) and (not isinstance(val, str)):
                for x in val:
                    check_result(x, depth=depth + 1)
            if isinstance(val, Mapping):
                for x in val.values():
                    check_result(x, depth=depth + 1)

        def check_iterator(it):
            for val in it:
                try:
                    check_result(val)
                except RuntimeError as exc:
                    raise RuntimeError(f'{self.name} returned a numpy scalar {val} ({type(val)})') from exc
                yield val
        if self.name in {'from_edgelist'}:
            pass
        elif isinstance(result, Iterator):
            result = check_iterator(result)
        else:
            try:
                check_result(result)
            except RuntimeError as exc:
                raise RuntimeError(f'{self.name} returned a numpy scalar {result} ({type(result)})') from exc
            check_result(result)
        if self.name in {'edmonds_karp', 'barycenter', 'contracted_edge', 'contracted_nodes', 'stochastic_graph', 'relabel_nodes', 'maximum_branching', 'incremental_closeness_centrality', 'minimal_branching', 'minimum_spanning_arborescence', 'recursive_simple_cycles', 'connected_double_edge_swap'}:
            bound = self.__signature__.bind(*converted_args, **converted_kwargs)
            bound.apply_defaults()
            bound2 = self.__signature__.bind(*args2, **kwargs2)
            bound2.apply_defaults()
            if self.name in {'minimal_branching', 'minimum_spanning_arborescence', 'recursive_simple_cycles', 'connected_double_edge_swap'}:
                G1 = backend.convert_to_nx(bound.arguments['G'])
                G2 = bound2.arguments['G']
                G2._adj = G1._adj
                nx._clear_cache(G2)
            elif self.name == 'edmonds_karp':
                R1 = backend.convert_to_nx(bound.arguments['residual'])
                R2 = bound2.arguments['residual']
                if R1 is not None and R2 is not None:
                    for k, v in R1.edges.items():
                        R2.edges[k]['flow'] = v['flow']
                    R2.graph.update(R1.graph)
                    nx._clear_cache(R2)
            elif self.name == 'barycenter' and bound.arguments['attr'] is not None:
                G1 = backend.convert_to_nx(bound.arguments['G'])
                G2 = bound2.arguments['G']
                attr = bound.arguments['attr']
                for k, v in G1.nodes.items():
                    G2.nodes[k][attr] = v[attr]
                nx._clear_cache(G2)
            elif self.name in {'contracted_nodes', 'contracted_edge'} and (not bound.arguments['copy']):
                G1 = backend.convert_to_nx(bound.arguments['G'])
                G2 = bound2.arguments['G']
                G2.__dict__.update(G1.__dict__)
                nx._clear_cache(G2)
            elif self.name == 'stochastic_graph' and (not bound.arguments['copy']):
                G1 = backend.convert_to_nx(bound.arguments['G'])
                G2 = bound2.arguments['G']
                for k, v in G1.edges.items():
                    G2.edges[k]['weight'] = v['weight']
                nx._clear_cache(G2)
            elif self.name == 'relabel_nodes' and (not bound.arguments['copy']) or self.name in {'incremental_closeness_centrality'}:
                G1 = backend.convert_to_nx(bound.arguments['G'])
                G2 = bound2.arguments['G']
                if G1 is G2:
                    return G2
                G2._node.clear()
                G2._node.update(G1._node)
                G2._adj.clear()
                G2._adj.update(G1._adj)
                if hasattr(G1, '_pred') and hasattr(G2, '_pred'):
                    G2._pred.clear()
                    G2._pred.update(G1._pred)
                if hasattr(G1, '_succ') and hasattr(G2, '_succ'):
                    G2._succ.clear()
                    G2._succ.update(G1._succ)
                nx._clear_cache(G2)
                if self.name == 'relabel_nodes':
                    return G2
            return backend.convert_to_nx(result)
        converted_result = backend.convert_to_nx(result)
        if isinstance(converted_result, nx.Graph) and self.name not in {'boykov_kolmogorov', 'preflow_push', 'quotient_graph', 'shortest_augmenting_path', 'spectral_graph_forge', 'read_gml', 'read_graph6', 'read_sparse6', 'bipartite_read_edgelist', 'read_adjlist', 'read_edgelist', 'read_graphml', 'read_multiline_adjlist', 'read_pajek', 'from_pydot', 'pydot_read_dot', 'agraph_read_dot', 'read_gexf'}:
            G = self.orig_func(*args2, **kwargs2)
            if not nx.utils.graphs_equal(G, converted_result):
                assert G.number_of_nodes() == converted_result.number_of_nodes()
                assert G.number_of_edges() == converted_result.number_of_edges()
                assert G.graph == converted_result.graph
                assert G.nodes == converted_result.nodes
                assert G.adj == converted_result.adj
                assert type(G) is type(converted_result)
                raise AssertionError('Graphs are not equal')
            return G
        return converted_result

    def _make_doc(self):
        """Generate the backends section at the end for functions having an alternate
        backend implementation(s) using the `backend_info` entry-point."""
        if not self.backends:
            return self._orig_doc
        lines = ['Backends', '--------']
        for backend in sorted(self.backends):
            info = backend_info[backend]
            if 'short_summary' in info:
                lines.append(f'{backend} : {info['short_summary']}')
            else:
                lines.append(backend)
            if 'functions' not in info or self.name not in info['functions']:
                lines.append('')
                continue
            func_info = info['functions'][self.name]
            if (func_docs := (func_info.get('additional_docs') or func_info.get('extra_docstring'))):
                lines.extend((f'  {line}' if line else line for line in func_docs.split('\n')))
                add_gap = True
            else:
                add_gap = False
            if (extra_parameters := (func_info.get('extra_parameters') or func_info.get('additional_parameters'))):
                if add_gap:
                    lines.append('')
                lines.append('  Additional parameters:')
                for param in sorted(extra_parameters):
                    lines.append(f'    {param}')
                    if (desc := extra_parameters[param]):
                        lines.append(f'      {desc}')
                    lines.append('')
            else:
                lines.append('')
            if (func_url := func_info.get('url')):
                lines.append(f'[`Source <{func_url}>`_]')
                lines.append('')
        lines.pop()
        to_add = '\n    '.join(lines)
        return f'{self._orig_doc.rstrip()}\n\n    {to_add}'

    def __reduce__(self):
        """Allow this object to be serialized with pickle.

        This uses the global registry `_registered_algorithms` to deserialize.
        """
        return (_restore_dispatchable, (self.name,))