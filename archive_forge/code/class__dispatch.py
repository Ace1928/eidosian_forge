import inspect
import os
import sys
import warnings
from functools import partial
from importlib.metadata import entry_points
from ..exception import NetworkXNotImplemented
class _dispatch:
    """Dispatches to a backend algorithm based on input graph types.

    Parameters
    ----------
    func : function

    name : str, optional
        The name of the algorithm to use for dispatching. If not provided,
        the name of ``func`` will be used. ``name`` is useful to avoid name
        conflicts, as all dispatched algorithms live in a single namespace.

    graphs : str or dict or None, default "G"
        If a string, the parameter name of the graph, which must be the first
        argument of the wrapped function. If more than one graph is required
        for the algorithm (or if the graph is not the first argument), provide
        a dict of parameter name to argument position for each graph argument.
        For example, ``@_dispatch(graphs={"G": 0, "auxiliary?": 4})``
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

    """
    _is_testing = False
    _fallback_to_nx = os.environ.get('NETWORKX_FALLBACK_TO_NX', 'true').strip().lower() == 'true'
    _automatic_backends = [x.strip() for x in os.environ.get('NETWORKX_AUTOMATIC_BACKENDS', '').split(',') if x.strip()]

    def __new__(cls, func=None, *, name=None, graphs='G', edge_attrs=None, node_attrs=None, preserve_edge_attrs=False, preserve_node_attrs=False, preserve_graph_attrs=False, preserve_all_attrs=False):
        if func is None:
            return partial(_dispatch, name=name, graphs=graphs, edge_attrs=edge_attrs, node_attrs=node_attrs, preserve_edge_attrs=preserve_edge_attrs, preserve_node_attrs=preserve_node_attrs, preserve_graph_attrs=preserve_graph_attrs, preserve_all_attrs=preserve_all_attrs)
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
        if edge_attrs is not None and (not isinstance(edge_attrs, (str, dict))):
            raise TypeError(f'Bad type for edge_attrs: {type(edge_attrs)}. Expected str or dict.') from None
        if node_attrs is not None and (not isinstance(node_attrs, (str, dict))):
            raise TypeError(f'Bad type for node_attrs: {type(node_attrs)}. Expected str or dict.') from None
        if not isinstance(self.preserve_edge_attrs, (bool, str, dict)):
            raise TypeError(f'Bad type for preserve_edge_attrs: {type(self.preserve_edge_attrs)}. Expected bool, str, or dict.') from None
        if not isinstance(self.preserve_node_attrs, (bool, str, dict)):
            raise TypeError(f'Bad type for preserve_node_attrs: {type(self.preserve_node_attrs)}. Expected bool, str, or dict.') from None
        if not isinstance(self.preserve_graph_attrs, (bool, set)):
            raise TypeError(f'Bad type for preserve_graph_attrs: {type(self.preserve_graph_attrs)}. Expected bool or set.') from None
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
        _registered_algorithms[name] = self
        return self

    @property
    def __doc__(self):
        if (rv := self._cached_doc) is not None:
            return rv
        rv = self._cached_doc = self._make_doc()
        return rv

    @__doc__.setter
    def __doc__(self, val):
        self._orig_doc = val
        self._cached_doc = None

    @property
    def __signature__(self):
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
        if self._is_testing and self._automatic_backends and (backend_name is None):
            return self._convert_and_call_for_tests(self._automatic_backends[0], args, kwargs, fallback_to_nx=self._fallback_to_nx)
        if self.list_graphs:
            args = list(args)
            for gname in self.list_graphs & graphs_resolved.keys():
                val = list(graphs_resolved[gname])
                graphs_resolved[gname] = val
                if gname in kwargs:
                    kwargs[gname] = val
                else:
                    args[self.graphs[gname]] = val
            has_backends = any((hasattr(g, '__networkx_backend__') or hasattr(g, '__networkx_plugin__') if gname not in self.list_graphs else any((hasattr(g2, '__networkx_backend__') or hasattr(g2, '__networkx_plugin__') for g2 in g)) for gname, g in graphs_resolved.items()))
            if has_backends:
                graph_backend_names = {getattr(g, '__networkx_backend__', getattr(g, '__networkx_plugin__', 'networkx')) for gname, g in graphs_resolved.items() if gname not in self.list_graphs}
                for gname in self.list_graphs & graphs_resolved.keys():
                    graph_backend_names.update((getattr(g, '__networkx_backend__', getattr(g, '__networkx_plugin__', 'networkx')) for g in graphs_resolved[gname]))
        else:
            has_backends = any((hasattr(g, '__networkx_backend__') or hasattr(g, '__networkx_plugin__') for g in graphs_resolved.values()))
            if has_backends:
                graph_backend_names = {getattr(g, '__networkx_backend__', getattr(g, '__networkx_plugin__', 'networkx')) for g in graphs_resolved.values()}
        if has_backends:
            backend_names = graph_backend_names - {'networkx'}
            if len(backend_names) != 1:
                raise TypeError(f'{self.name}() graphs must all be from the same backend, found {backend_names}')
            [graph_backend_name] = backend_names
            if backend_name is not None and backend_name != graph_backend_name:
                raise TypeError(f'{self.name}() is unable to convert graph from backend {graph_backend_name!r} to the specified backend {backend_name!r}.')
            if graph_backend_name not in backends:
                raise ImportError(f'Unable to load backend: {graph_backend_name}')
            if 'networkx' in graph_backend_names and graph_backend_name not in self._automatic_backends:
                raise TypeError(f'Unable to convert inputs and run {self.name}. {self.name}() has networkx and {graph_backend_name} graphs, but NetworkX is not configured to automatically convert graphs from networkx to {graph_backend_name}.')
            backend = _load_backend(graph_backend_name)
            if hasattr(backend, self.name):
                if 'networkx' in graph_backend_names:
                    return self._convert_and_call(graph_backend_name, args, kwargs, fallback_to_nx=self._fallback_to_nx)
                return getattr(backend, self.name)(*args, **kwargs)
            raise NetworkXNotImplemented(f"'{self.name}' not implemented by {graph_backend_name}")
        if backend_name is not None:
            return self._convert_and_call(backend_name, args, kwargs, fallback_to_nx=False)
        if self.graphs:
            for backend_name in self._automatic_backends:
                if self._can_backend_run(backend_name, *args, **kwargs):
                    return self._convert_and_call(backend_name, args, kwargs, fallback_to_nx=self._fallback_to_nx)
        return self.orig_func(*args, **kwargs)

    def _can_backend_run(self, backend_name, /, *args, **kwargs):
        """Can the specified backend run this algorithms with these arguments?"""
        backend = _load_backend(backend_name)
        return hasattr(backend, self.name) and (not hasattr(backend, 'can_run') or backend.can_run(self.name, args, kwargs))

    def _convert_arguments(self, backend_name, args, kwargs):
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
        backend = _load_backend(backend_name)
        for gname in self.graphs:
            if gname in self.list_graphs:
                bound.arguments[gname] = [backend.convert_from_nx(g, edge_attrs=edge_attrs, node_attrs=node_attrs, preserve_edge_attrs=preserve_edge_attrs, preserve_node_attrs=preserve_node_attrs, preserve_graph_attrs=preserve_graph_attrs, name=self.name, graph_name=gname) if getattr(g, '__networkx_backend__', getattr(g, '__networkx_plugin__', 'networkx')) == 'networkx' else g for g in bound.arguments[gname]]
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
                if getattr(graph, '__networkx_backend__', getattr(graph, '__networkx_plugin__', 'networkx')) == 'networkx':
                    bound.arguments[gname] = backend.convert_from_nx(graph, edge_attrs=edges, node_attrs=nodes, preserve_edge_attrs=preserve_edges, preserve_node_attrs=preserve_nodes, preserve_graph_attrs=preserve_graph, name=self.name, graph_name=gname)
        bound_kwargs = bound.kwargs
        del bound_kwargs['backend']
        return (bound.args, bound_kwargs)

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
            converted_args, converted_kwargs = self._convert_arguments(backend_name, args, kwargs)
            result = getattr(backend, self.name)(*converted_args, **converted_kwargs)
        except (NotImplementedError, NetworkXNotImplemented) as exc:
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
        try:
            converted_args, converted_kwargs = self._convert_arguments(backend_name, args, kwargs)
            result = getattr(backend, self.name)(*converted_args, **converted_kwargs)
        except (NotImplementedError, NetworkXNotImplemented) as exc:
            if fallback_to_nx:
                return self.orig_func(*args, **kwargs)
            import pytest
            pytest.xfail(exc.args[0] if exc.args else f'{self.name} raised {type(exc).__name__}')
        if self.name in {'edmonds_karp_core', 'barycenter', 'contracted_nodes', 'stochastic_graph', 'relabel_nodes'}:
            bound = self.__signature__.bind(*converted_args, **converted_kwargs)
            bound.apply_defaults()
            bound2 = self.__signature__.bind(*args, **kwargs)
            bound2.apply_defaults()
            if self.name == 'edmonds_karp_core':
                R1 = backend.convert_to_nx(bound.arguments['R'])
                R2 = bound2.arguments['R']
                for k, v in R1.edges.items():
                    R2.edges[k]['flow'] = v['flow']
            elif self.name == 'barycenter' and bound.arguments['attr'] is not None:
                G1 = backend.convert_to_nx(bound.arguments['G'])
                G2 = bound2.arguments['G']
                attr = bound.arguments['attr']
                for k, v in G1.nodes.items():
                    G2.nodes[k][attr] = v[attr]
            elif self.name == 'contracted_nodes' and (not bound.arguments['copy']):
                G1 = backend.convert_to_nx(bound.arguments['G'])
                G2 = bound2.arguments['G']
                G2.__dict__.update(G1.__dict__)
            elif self.name == 'stochastic_graph' and (not bound.arguments['copy']):
                G1 = backend.convert_to_nx(bound.arguments['G'])
                G2 = bound2.arguments['G']
                for k, v in G1.edges.items():
                    G2.edges[k]['weight'] = v['weight']
            elif self.name == 'relabel_nodes' and (not bound.arguments['copy']):
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
                return G2
        return backend.convert_to_nx(result, name=self.name)

    def _make_doc(self):
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
            if 'extra_docstring' in func_info:
                lines.extend((f'  {line}' if line else line for line in func_info['extra_docstring'].split('\n')))
                add_gap = True
            else:
                add_gap = False
            if 'extra_parameters' in func_info:
                if add_gap:
                    lines.append('')
                lines.append('  Extra parameters:')
                extra_parameters = func_info['extra_parameters']
                for param in sorted(extra_parameters):
                    lines.append(f'    {param}')
                    if (desc := extra_parameters[param]):
                        lines.append(f'      {desc}')
                    lines.append('')
            else:
                lines.append('')
        lines.pop()
        to_add = '\n    '.join(lines)
        return f'{self._orig_doc.rstrip()}\n\n    {to_add}'

    def __reduce__(self):
        """Allow this object to be serialized with pickle.

        This uses the global registry `_registered_algorithms` to deserialize.
        """
        return (_restore_dispatch, (self.name,))