from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Any, Dict, List
Converts the given graph into a graph_tool.Graph().

    The subjects and objects are the later vertices of the Graph.
    The predicates become edges.

    :Parameters:
        - graph: a rdflib.Graph.
        - v_prop_names: a list of names for the vertex properties. The default is set
          to ['term'] (see transform_s, transform_o below).
        - e_prop_names: a list of names for the edge properties.
        - transform_s: callable with s, p, o input. Should return a dictionary
          containing a value for each name in v_prop_names. By default is set
          to {'term': s} which in combination with v_prop_names = ['term']
          adds s as 'term' property to the generated vertex for s.
        - transform_p: similar to transform_s, but wrt. e_prop_names. By default
          returns {'term': p} which adds p as a property to the generated
          edge between the vertex for s and the vertex for o.
        - transform_o: similar to transform_s.

    Returns: graph_tool.Graph()

    >>> from rdflib import Graph, URIRef, Literal
    >>> g = Graph()
    >>> a, b, l = URIRef('a'), URIRef('b'), Literal('l')
    >>> p, q = URIRef('p'), URIRef('q')
    >>> edges = [(a, p, b), (a, q, b), (b, p, a), (b, p, l)]
    >>> for t in edges:
    ...     g.add(t)
    ...
    >>> mdg = rdflib_to_graphtool(g)
    >>> len(list(mdg.edges()))
    4
    >>> from graph_tool import util as gt_util
    >>> vpterm = mdg.vertex_properties['term']
    >>> va = gt_util.find_vertex(mdg, vpterm, a)[0]
    >>> vb = gt_util.find_vertex(mdg, vpterm, b)[0]
    >>> vl = gt_util.find_vertex(mdg, vpterm, l)[0]
    >>> (va, vb) in [(e.source(), e.target()) for e in list(mdg.edges())]
    True
    >>> epterm = mdg.edge_properties['term']
    >>> len(list(gt_util.find_edge(mdg, epterm, p))) == 3
    True
    >>> len(list(gt_util.find_edge(mdg, epterm, q))) == 1
    True

    >>> mdg = rdflib_to_graphtool(
    ...     g,
    ...     e_prop_names=[str('name')],
    ...     transform_p=lambda s, p, o: {str('name'): unicode(p)})
    >>> epterm = mdg.edge_properties['name']
    >>> len(list(gt_util.find_edge(mdg, epterm, unicode(p)))) == 3
    True
    >>> len(list(gt_util.find_edge(mdg, epterm, unicode(q)))) == 1
    True

    