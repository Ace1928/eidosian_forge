from __future__ import annotations
import abc
import copy
import html
from collections.abc import (
from typing import Any
import tlz as toolz
import dask
from dask import config
from dask.base import clone_key, flatten, is_dask_collection, normalize_token
from dask.core import keys_in_tasks, reverse_dict
from dask.typing import DaskCollection, Graph, Key
from dask.utils import ensure_dict, import_required, key_split
from dask.widgets import get_template
class HighLevelGraph(Graph):
    """Task graph composed of layers of dependent subgraphs

    This object encodes a Dask task graph that is composed of layers of
    dependent subgraphs, such as commonly occurs when building task graphs
    using high level collections like Dask array, bag, or dataframe.

    Typically each high level array, bag, or dataframe operation takes the task
    graphs of the input collections, merges them, and then adds one or more new
    layers of tasks for the new operation.  These layers typically have at
    least as many tasks as there are partitions or chunks in the collection.
    The HighLevelGraph object stores the subgraphs for each operation
    separately in sub-graphs, and also stores the dependency structure between
    them.

    Parameters
    ----------
    layers : Mapping[str, Mapping]
        The subgraph layers, keyed by a unique name
    dependencies : Mapping[str, set[str]]
        The set of layers on which each layer depends
    key_dependencies : dict[Key, set], optional
        Mapping (some) keys in the high level graph to their dependencies. If
        a key is missing, its dependencies will be calculated on-the-fly.

    Examples
    --------
    Here is an idealized example that shows the internal state of a
    HighLevelGraph

    >>> import dask.dataframe as dd

    >>> df = dd.read_csv('myfile.*.csv')  # doctest: +SKIP
    >>> df = df + 100  # doctest: +SKIP
    >>> df = df[df.name == 'Alice']  # doctest: +SKIP

    >>> graph = df.__dask_graph__()  # doctest: +SKIP
    >>> graph.layers  # doctest: +SKIP
    {
     'read-csv': {('read-csv', 0): (pandas.read_csv, 'myfile.0.csv'),
                  ('read-csv', 1): (pandas.read_csv, 'myfile.1.csv'),
                  ('read-csv', 2): (pandas.read_csv, 'myfile.2.csv'),
                  ('read-csv', 3): (pandas.read_csv, 'myfile.3.csv')},
     'add': {('add', 0): (operator.add, ('read-csv', 0), 100),
             ('add', 1): (operator.add, ('read-csv', 1), 100),
             ('add', 2): (operator.add, ('read-csv', 2), 100),
             ('add', 3): (operator.add, ('read-csv', 3), 100)}
     'filter': {('filter', 0): (lambda part: part[part.name == 'Alice'], ('add', 0)),
                ('filter', 1): (lambda part: part[part.name == 'Alice'], ('add', 1)),
                ('filter', 2): (lambda part: part[part.name == 'Alice'], ('add', 2)),
                ('filter', 3): (lambda part: part[part.name == 'Alice'], ('add', 3))}
    }

    >>> graph.dependencies  # doctest: +SKIP
    {
     'read-csv': set(),
     'add': {'read-csv'},
     'filter': {'add'}
    }

    See Also
    --------
    HighLevelGraph.from_collections :
        typically used by developers to make new HighLevelGraphs
    """
    layers: Mapping[str, Layer]
    dependencies: Mapping[str, set[str]]
    key_dependencies: dict[Key, set[Key]]
    _to_dict: dict
    _all_external_keys: set

    def __init__(self, layers: Mapping[str, Graph], dependencies: Mapping[str, set[str]], key_dependencies: dict[Key, set[Key]] | None=None):
        self.dependencies = dependencies
        self.key_dependencies = key_dependencies or {}
        self.layers = {k: v if isinstance(v, Layer) else MaterializedLayer(v) for k, v in layers.items()}

    @classmethod
    def _from_collection(cls, name, layer, collection):
        """`from_collections` optimized for a single collection"""
        if not is_dask_collection(collection):
            raise TypeError(type(collection))
        graph = collection.__dask_graph__()
        if isinstance(graph, HighLevelGraph):
            layers = ensure_dict(graph.layers, copy=True)
            layers[name] = layer
            deps = ensure_dict(graph.dependencies, copy=True)
            deps[name] = set(collection.__dask_layers__())
        else:
            key = _get_some_layer_name(collection)
            layers = {name: layer, key: graph}
            deps = {name: {key}, key: set()}
        return cls(layers, deps)

    @classmethod
    def from_collections(cls, name: str, layer: Graph, dependencies: Sequence[DaskCollection]=()) -> HighLevelGraph:
        """Construct a HighLevelGraph from a new layer and a set of collections

        This constructs a HighLevelGraph in the common case where we have a single
        new layer and a set of old collections on which we want to depend.

        This pulls out the ``__dask_layers__()`` method of the collections if
        they exist, and adds them to the dependencies for this new layer.  It
        also merges all of the layers from all of the dependent collections
        together into the new layers for this graph.

        Parameters
        ----------
        name : str
            The name of the new layer
        layer : Mapping
            The graph layer itself
        dependencies : List of Dask collections
            A list of other dask collections (like arrays or dataframes) that
            have graphs themselves

        Examples
        --------

        In typical usage we make a new task layer, and then pass that layer
        along with all dependent collections to this method.

        >>> def add(self, other):
        ...     name = 'add-' + tokenize(self, other)
        ...     layer = {(name, i): (add, input_key, other)
        ...              for i, input_key in enumerate(self.__dask_keys__())}
        ...     graph = HighLevelGraph.from_collections(name, layer, dependencies=[self])
        ...     return new_collection(name, graph)
        """
        if len(dependencies) == 1:
            return cls._from_collection(name, layer, dependencies[0])
        layers = {name: layer}
        name_dep: set[str] = set()
        deps: dict[str, set[str]] = {name: name_dep}
        for collection in toolz.unique(dependencies, key=id):
            if is_dask_collection(collection):
                graph = collection.__dask_graph__()
                if isinstance(graph, HighLevelGraph):
                    layers.update(graph.layers)
                    deps.update(graph.dependencies)
                    name_dep |= set(collection.__dask_layers__())
                else:
                    key = _get_some_layer_name(collection)
                    layers[key] = graph
                    name_dep.add(key)
                    deps[key] = set()
            else:
                raise TypeError(type(collection))
        return cls(layers, deps)

    def __getitem__(self, key: Key) -> Any:
        try:
            return self.layers[key][key]
        except KeyError:
            pass
        try:
            return self.layers[key[0]][key]
        except (KeyError, IndexError, TypeError):
            pass
        for d in self.layers.values():
            try:
                return d[key]
            except KeyError:
                pass
        raise KeyError(key)

    def __len__(self) -> int:
        return sum((len(layer) for layer in self.layers.values()))

    def __iter__(self) -> Iterator[Key]:
        return iter(self.to_dict())

    def to_dict(self) -> dict[Key, Any]:
        """Efficiently convert to plain dict. This method is faster than dict(self)."""
        try:
            return self._to_dict
        except AttributeError:
            out = self._to_dict = ensure_dict(self)
            return out

    def keys(self) -> KeysView:
        """Get all keys of all the layers.

        This will in many cases materialize layers, which makes it a relatively
        expensive operation. See :meth:`get_all_external_keys` for a faster alternative.
        """
        return self.to_dict().keys()

    def get_all_external_keys(self) -> set[Key]:
        """Get all output keys of all layers

        This will in most cases _not_ materialize any layers, which makes
        it a relative cheap operation.

        Returns
        -------
        keys: set
            A set of all external keys
        """
        try:
            return self._all_external_keys
        except AttributeError:
            keys: set = set()
            for layer in self.layers.values():
                keys.update(layer.get_output_keys())
            self._all_external_keys = keys
            return keys

    def items(self) -> ItemsView[Key, Any]:
        return self.to_dict().items()

    def values(self) -> ValuesView[Any]:
        return self.to_dict().values()

    def get_all_dependencies(self) -> dict[Key, set[Key]]:
        """Get dependencies of all keys

        This will in most cases materialize all layers, which makes
        it an expensive operation.

        Returns
        -------
        map: Mapping
            A map that maps each key to its dependencies
        """
        all_keys = self.keys()
        missing_keys = all_keys - self.key_dependencies.keys()
        if missing_keys:
            for layer in self.layers.values():
                for k in missing_keys & layer.keys():
                    self.key_dependencies[k] = layer.get_dependencies(k, all_keys)
        return self.key_dependencies

    @property
    def dependents(self) -> dict[str, set[str]]:
        return reverse_dict(self.dependencies)

    def copy(self) -> HighLevelGraph:
        return HighLevelGraph(ensure_dict(self.layers, copy=True), ensure_dict(self.dependencies, copy=True), self.key_dependencies.copy())

    @classmethod
    def merge(cls, *graphs: Graph) -> HighLevelGraph:
        layers: dict[str, Graph] = {}
        dependencies: dict[str, set[str]] = {}
        for g in graphs:
            if isinstance(g, HighLevelGraph):
                layers.update(g.layers)
                dependencies.update(g.dependencies)
            elif isinstance(g, Mapping):
                layers[str(id(g))] = g
                dependencies[str(id(g))] = set()
            else:
                raise TypeError(g)
        return cls(layers, dependencies)

    def visualize(self, filename='dask-hlg.svg', format=None, **kwargs):
        """
        Visualize this dask high level graph.

        Requires ``graphviz`` to be installed.

        Parameters
        ----------
        filename : str or None, optional
            The name of the file to write to disk. If the provided `filename`
            doesn't include an extension, '.png' will be used by default.
            If `filename` is None, no file will be written, and the graph is
            rendered in the Jupyter notebook only.
        format : {'png', 'pdf', 'dot', 'svg', 'jpeg', 'jpg'}, optional
            Format in which to write output file. Default is 'svg'.
        color : {None, 'layer_type'}, optional (default: None)
            Options to color nodes.
            - None, no colors.
            - layer_type, color nodes based on the layer type.
        **kwargs
           Additional keyword arguments to forward to ``to_graphviz``.

        Examples
        --------
        >>> x.dask.visualize(filename='dask.svg')  # doctest: +SKIP
        >>> x.dask.visualize(filename='dask.svg', color='layer_type')  # doctest: +SKIP

        Returns
        -------
        result : IPython.display.Image, IPython.display.SVG, or None
            See dask.dot.dot_graph for more information.

        See Also
        --------
        dask.dot.dot_graph
        dask.base.visualize # low level variant
        """
        from dask.dot import graphviz_to_file
        g = to_graphviz(self, **kwargs)
        graphviz_to_file(g, filename, format)
        return g

    def _toposort_layers(self) -> list[str]:
        """Sort the layers in a high level graph topologically

        Parameters
        ----------
        hlg : HighLevelGraph
            The high level graph's layers to sort

        Returns
        -------
        sorted: list
            List of layer names sorted topologically
        """
        degree = {k: len(v) for k, v in self.dependencies.items()}
        reverse_deps: dict[str, list[str]] = {k: [] for k in self.dependencies}
        ready = []
        for k, v in self.dependencies.items():
            for dep in v:
                reverse_deps[dep].append(k)
            if not v:
                ready.append(k)
        ret = []
        while len(ready) > 0:
            layer = ready.pop()
            ret.append(layer)
            for rdep in reverse_deps[layer]:
                degree[rdep] -= 1
                if degree[rdep] == 0:
                    ready.append(rdep)
        return ret

    def cull(self, keys: Iterable[Key]) -> HighLevelGraph:
        """Return new HighLevelGraph with only the tasks required to calculate keys.

        In other words, remove unnecessary tasks from dask.

        Parameters
        ----------
        keys
            iterable of keys or nested list of keys such as the output of
            ``__dask_keys__()``

        Returns
        -------
        hlg: HighLevelGraph
            Culled high level graph
        """
        from dask.layers import Blockwise
        keys_set = set(flatten(keys))
        all_ext_keys = self.get_all_external_keys()
        ret_layers: dict = {}
        ret_key_deps: dict = {}
        for layer_name in reversed(self._toposort_layers()):
            layer = self.layers[layer_name]
            output_keys = keys_set.intersection(layer.get_output_keys())
            if output_keys:
                culled_layer, culled_deps = layer.cull(output_keys, all_ext_keys)
                external_deps = set()
                for d in culled_deps.values():
                    external_deps |= d
                external_deps -= culled_layer.get_output_keys()
                keys_set |= external_deps
                ret_layers[layer_name] = culled_layer
                if isinstance(layer, Blockwise) or isinstance(layer, MaterializedLayer) or (layer.is_materialized() and len(layer) == len(culled_deps)):
                    ret_key_deps.update(culled_deps)
        ret_layers_keys = set(ret_layers.keys())
        ret_dependencies = {layer_name: self.dependencies[layer_name] & ret_layers_keys for layer_name in ret_layers}
        return HighLevelGraph(ret_layers, ret_dependencies, ret_key_deps)

    def cull_layers(self, layers: Iterable[str]) -> HighLevelGraph:
        """Return a new HighLevelGraph with only the given layers and their
        dependencies. Internally, layers are not modified.

        This is a variant of :meth:`HighLevelGraph.cull` which is much faster and does
        not risk creating a collision between two layers with the same name and
        different content when two culled graphs are merged later on.

        Returns
        -------
        hlg: HighLevelGraph
            Culled high level graph
        """
        to_visit = set(layers)
        ret_layers = {}
        ret_dependencies = {}
        while to_visit:
            k = to_visit.pop()
            ret_layers[k] = self.layers[k]
            ret_dependencies[k] = self.dependencies[k]
            to_visit |= ret_dependencies[k] - ret_dependencies.keys()
        return HighLevelGraph(ret_layers, ret_dependencies)

    def validate(self) -> None:
        for layer_name, deps in self.dependencies.items():
            if layer_name not in self.layers:
                raise ValueError(f'dependencies[{repr(layer_name)}] not found in layers')
            for dep in deps:
                if dep not in self.dependencies:
                    raise ValueError(f'{repr(dep)} not found in dependencies')
        for layer in self.layers.values():
            assert hasattr(layer, 'annotations')
        dependencies = compute_layer_dependencies(self.layers)
        dep_key1 = self.dependencies.keys()
        dep_key2 = dependencies.keys()
        if dep_key1 != dep_key2:
            raise ValueError(f'incorrect dependencies keys {set(dep_key1)!r} expected {set(dep_key2)!r}')
        for k in dep_key1:
            if self.dependencies[k] != dependencies[k]:
                raise ValueError(f'incorrect dependencies[{repr(k)}]: {repr(self.dependencies[k])} expected {repr(dependencies[k])}')

    def __repr__(self) -> str:
        representation = f'{type(self).__name__} with {len(self.layers)} layers.\n'
        representation += f'<{self.__class__.__module__}.{self.__class__.__name__} object at {hex(id(self))}>\n'
        for i, layerkey in enumerate(self._toposort_layers()):
            representation += f' {i}. {layerkey}\n'
        return representation

    def _repr_html_(self) -> str:
        return get_template('highlevelgraph.html.j2').render(type=type(self).__name__, layers=self.layers, toposort=self._toposort_layers(), layer_dependencies=self.dependencies, n_outputs=len(self.get_all_external_keys()))