import re
import itertools
import os
import logging
import string
import pyparsing
from pyparsing import __version__ as pyparsing_version
from pyparsing import (Literal, CaselessLiteral, Word, OneOrMore, Forward, Group, Optional, Combine, restOfLine,
from collections import OrderedDict
def _proc_edge_stmt(self, toks):
    """Return (ADD_EDGE, src, dest, options)"""
    edgelist = []
    opts = toks[-1]
    if not isinstance(opts, dict):
        opts = {}
    for src, op, dest in windows(toks, length=3, overlap=1, padding=False):
        srcgraph = destgraph = False
        if len(src) > 1 and src[0] == ADD_SUBGRAPH:
            edgelist.append(src)
            srcgraph = True
        if len(dest) > 1 and dest[0] == ADD_SUBGRAPH:
            edgelist.append(dest)
            destgraph = True
        if srcgraph or destgraph:
            if srcgraph and destgraph:
                edgelist.append((ADD_GRAPH_TO_GRAPH_EDGE, src[1], dest[1], opts))
            elif srcgraph:
                edgelist.append((ADD_GRAPH_TO_NODE_EDGE, src[1], dest, opts))
            else:
                edgelist.append((ADD_NODE_TO_GRAPH_EDGE, src, dest[1], opts))
        else:
            edgelist.append((ADD_EDGE, src, dest, opts))
    return edgelist