import collections
import os
import typing
from dataclasses import dataclass
class NetworkXConfig(Config):
    """Configuration for NetworkX that controls behaviors such as how to use backends.

    Attribute and bracket notation are supported for getting and setting configurations:

    >>> nx.config.backend_priority == nx.config["backend_priority"]
    True

    Parameters
    ----------
    backend_priority : list of backend names
        Enable automatic conversion of graphs to backend graphs for algorithms
        implemented by the backend. Priority is given to backends listed earlier.
        Default is empty list.

    backends : Config mapping of backend names to backend Config
        The keys of the Config mapping are names of all installed NetworkX backends,
        and the values are their configurations as Config mappings.

    cache_converted_graphs : bool
        If True, then save converted graphs to the cache of the input graph. Graph
        conversion may occur when automatically using a backend from `backend_priority`
        or when using the `backend=` keyword argument to a function call. Caching can
        improve performance by avoiding repeated conversions, but it uses more memory.
        Care should be taken to not manually mutate a graph that has cached graphs; for
        example, ``G[u][v][k] = val`` changes the graph, but does not clear the cache.
        Using methods such as ``G.add_edge(u, v, weight=val)`` will clear the cache to
        keep it consistent. ``G.__networkx_cache__.clear()`` manually clears the cache.
        Default is False.

    Notes
    -----
    Environment variables may be used to control some default configurations:

    - NETWORKX_BACKEND_PRIORITY: set `backend_priority` from comma-separated names.
    - NETWORKX_CACHE_CONVERTED_GRAPHS: set `cache_converted_graphs` to True if nonempty.

    This is a global configuration. Use with caution when using from multiple threads.
    """
    backend_priority: list[str]
    backends: Config
    cache_converted_graphs: bool

    def _check_config(self, key, value):
        from .backends import backends
        if key == 'backend_priority':
            if not (isinstance(value, list) and all((isinstance(x, str) for x in value))):
                raise TypeError(f'{key!r} config must be a list of backend names; got {value!r}')
            if (missing := {x for x in value if x not in backends}):
                missing = ', '.join(map(repr, sorted(missing)))
                raise ValueError(f'Unknown backend when setting {key!r}: {missing}')
        elif key == 'backends':
            if not (isinstance(value, Config) and all((isinstance(key, str) for key in value)) and all((isinstance(val, Config) for val in value.values()))):
                raise TypeError(f'{key!r} config must be a Config of backend configs; got {value!r}')
            if (missing := {x for x in value if x not in backends}):
                missing = ', '.join(map(repr, sorted(missing)))
                raise ValueError(f'Unknown backend when setting {key!r}: {missing}')
        elif key == 'cache_converted_graphs':
            if not isinstance(value, bool):
                raise TypeError(f'{key!r} config must be True or False; got {value!r}')