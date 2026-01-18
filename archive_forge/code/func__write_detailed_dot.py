import os
import sys
import pickle
from collections import defaultdict
import re
from copy import deepcopy
from glob import glob
from pathlib import Path
from traceback import format_exception
from hashlib import sha1
from functools import reduce
import numpy as np
from ... import logging, config
from ...utils.filemanip import (
from ...utils.misc import str2bool
from ...utils.functions import create_function_from_source
from ...interfaces.base.traits_extension import (
from ...interfaces.base.support import Bunch, InterfaceResult
from ...interfaces.base import CommandLine
from ...interfaces.utility import IdentityInterface
from ...utils.provenance import ProvStore, pm, nipype_ns, get_id
from inspect import signature
def _write_detailed_dot(graph, dotfilename):
    """
    Create a dot file with connection info ::

        digraph structs {
        node [shape=record];
        struct1 [label="<f0> left|<f1> middle|<f2> right"];
        struct2 [label="<f0> one|<f1> two"];
        struct3 [label="hello\\nworld |{ b |{c|<here> d|e}| f}| g | h"];
        struct1:f1 -> struct2:f0;
        struct1:f0 -> struct2:f1;
        struct1:f2 -> struct3:here;
        }
    """
    import networkx as nx
    text = ['digraph structs {', 'node [shape=record];']
    edges = []
    for n in nx.topological_sort(graph):
        nodename = n.itername
        inports = []
        for u, v, d in graph.in_edges(nbunch=n, data=True):
            for cd in d['connect']:
                if isinstance(cd[0], (str, bytes)):
                    outport = cd[0]
                else:
                    outport = cd[0][0]
                inport = cd[1]
                ipstrip = 'in%s' % _replacefunk(inport)
                opstrip = 'out%s' % _replacefunk(outport)
                edges.append('%s:%s:e -> %s:%s:w;' % (u.itername.replace('.', ''), opstrip, v.itername.replace('.', ''), ipstrip))
                if inport not in inports:
                    inports.append(inport)
        inputstr = ['{IN'] + ['|<in%s> %s' % (_replacefunk(ip), ip) for ip in sorted(inports)] + ['}']
        outports = []
        for u, v, d in graph.out_edges(nbunch=n, data=True):
            for cd in d['connect']:
                if isinstance(cd[0], (str, bytes)):
                    outport = cd[0]
                else:
                    outport = cd[0][0]
                if outport not in outports:
                    outports.append(outport)
        outputstr = ['{OUT'] + ['|<out%s> %s' % (_replacefunk(oport), oport) for oport in sorted(outports)] + ['}']
        srcpackage = ''
        if hasattr(n, '_interface'):
            pkglist = n.interface.__class__.__module__.split('.')
            if len(pkglist) > 2:
                srcpackage = pkglist[2]
        srchierarchy = '.'.join(nodename.split('.')[1:-1])
        nodenamestr = '{ %s | %s | %s }' % (nodename.split('.')[-1], srcpackage, srchierarchy)
        text += ['%s [label="%s|%s|%s"];' % (nodename.replace('.', ''), ''.join(inputstr), nodenamestr, ''.join(outputstr))]
    for edge in sorted(edges):
        text.append(edge)
    text.append('}')
    with open(dotfilename, 'wt') as filep:
        filep.write('\n'.join(text))
    return text