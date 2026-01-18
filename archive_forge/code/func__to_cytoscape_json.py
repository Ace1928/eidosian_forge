from __future__ import annotations
import os
import re
from functools import partial
from dask.core import get_dependencies, ishashable, istask
from dask.utils import apply, funcname, import_required, key_split
def _to_cytoscape_json(dsk, data_attributes=None, function_attributes=None, collapse_outputs=False, verbose=False, **kwargs):
    """
    Convert a dask graph to Cytoscape JSON:
    https://js.cytoscape.org/#notation/elements-json
    """
    nodes = []
    edges = []
    data = {'nodes': nodes, 'edges': edges}
    data_attributes = data_attributes or {}
    function_attributes = function_attributes or {}
    seen = set()
    connected = set()
    for k, v in dsk.items():
        k_name = name(k)
        if istask(v):
            func_name = name((k, 'function')) if not collapse_outputs else k_name
            if collapse_outputs or func_name not in seen:
                seen.add(func_name)
                attrs = function_attributes.get(k, {}).copy()
                nodes.append({'data': {'id': func_name, 'label': key_split(k), 'shape': 'ellipse', 'color': 'gray', **attrs}})
            if not collapse_outputs:
                edges.append({'data': {'source': func_name, 'target': k_name}})
                connected.add(func_name)
                connected.add(k_name)
            for dep in get_dependencies(dsk, k):
                dep_name = name(dep)
                if dep_name not in seen:
                    seen.add(dep_name)
                    attrs = data_attributes.get(dep, {}).copy()
                    nodes.append({'data': {'id': dep_name, 'label': box_label(dep, verbose), 'shape': 'rectangle', 'color': 'gray', **attrs}})
                edges.append({'data': {'source': dep_name, 'target': func_name}})
                connected.add(dep_name)
                connected.add(func_name)
        elif ishashable(v) and v in dsk:
            v_name = name(v)
            edges.append({'data': {'source': v_name, 'target': k_name}})
            connected.add(v_name)
            connected.add(k_name)
        if (not collapse_outputs or k_name in connected) and k_name not in seen:
            seen.add(k_name)
            attrs = data_attributes.get(k, {}).copy()
            nodes.append({'data': {'id': k_name, 'label': box_label(k, verbose), 'shape': 'rectangle', 'color': 'gray', **attrs}})
    return data