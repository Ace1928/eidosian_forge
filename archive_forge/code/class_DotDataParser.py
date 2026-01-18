import re
import itertools
import os
import logging
import string
import pyparsing
from pyparsing import __version__ as pyparsing_version
from pyparsing import (Literal, CaselessLiteral, Word, OneOrMore, Forward, Group, Optional, Combine, restOfLine,
from collections import OrderedDict
class DotDataParser(object):
    """Container class for parsing Graphviz dot data"""

    def __init__(self):
        pass
        self.dotparser = self.define_dot_parser()

    def _proc_node_id(self, toks):
        if len(toks) > 1:
            return (toks[0], toks[1])
        else:
            return toks

    def _proc_attr_list(self, toks):
        return dict(nsplit(toks, 2))

    def _proc_attr_list_combine(self, toks):
        if toks:
            first_dict = toks[0]
            for d in toks:
                first_dict.update(d)
            return first_dict
        return toks

    def _proc_attr_assignment(self, toks):
        return (SET_GRAPH_ATTR, dict(nsplit(toks, 2)))

    def _proc_node_stmt(self, toks):
        """Return (ADD_NODE, node_name, options)"""
        if len(toks) == 2:
            return tuple([ADD_NODE] + list(toks))
        else:
            return tuple([ADD_NODE] + list(toks) + [{}])

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

    def _proc_default_attr_stmt(self, toks):
        """Return (ADD_DEFAULT_NODE_ATTR,options"""
        if len(toks) == 1:
            gtype = toks
            attr = {}
        else:
            gtype, attr = toks
        if gtype == 'node':
            return (SET_DEF_NODE_ATTR, attr)
        elif gtype == 'edge':
            return (SET_DEF_EDGE_ATTR, attr)
        elif gtype == 'graph':
            return (SET_DEF_GRAPH_ATTR, attr)
        else:
            return ('unknown', toks)

    def _proc_subgraph_stmt(self, toks):
        """Returns (ADD_SUBGRAPH, name, elements)"""
        return ('add_subgraph', toks[1], toks[2].asList())

    def _main_graph_stmt(self, toks):
        return (toks[0], toks[1], toks[2], toks[3].asList())

    def define_dot_parser(self):
        """Define dot grammar

        Based on the grammar http://www.graphviz.org/doc/info/lang.html
        """
        colon = Literal(':')
        lbrace = Suppress('{')
        rbrace = Suppress('}')
        lbrack = Suppress('[')
        rbrack = Suppress(']')
        lparen = Literal('(')
        rparen = Literal(')')
        equals = Suppress('=')
        comma = Literal(',')
        dot = Literal('.')
        slash = Literal('/')
        bslash = Literal('\\')
        star = Literal('*')
        semi = Suppress(';')
        at = Literal('@')
        minus = Literal('-')
        pluss = Suppress('+')
        strict_ = CaselessLiteral('strict')
        graph_ = CaselessLiteral('graph')
        digraph_ = CaselessLiteral('digraph')
        subgraph_ = CaselessLiteral('subgraph')
        node_ = CaselessLiteral('node')
        edge_ = CaselessLiteral('edge')
        punctuation_ = ''.join([c for c in string.punctuation if c not in '_']) + string.whitespace
        identifier = Word(alphanums + '_').setName('identifier')
        double_quoted_string = Regex('\\"(?:\\\\\\"|\\\\\\\\|[^"])*\\"', re.MULTILINE)
        double_quoted_string.setParseAction(removeQuotes)
        quoted_string = Combine(double_quoted_string + Optional(OneOrMore(pluss + double_quoted_string)), adjacent=False)
        alphastring_ = OneOrMore(CharsNotIn(punctuation_))

        def parse_html(s, loc, toks):
            return '<<%s>>' % ''.join(toks[0])
        opener = '<'
        closer = '>'
        try:
            html_text = pyparsing.nestedExpr(opener, closer, CharsNotIn(opener + closer).setParseAction(lambda t: t[0])).setParseAction(parse_html)
        except:
            log.debug('nestedExpr not available.')
            log.warning('Old version of pyparsing detected. Version 1.4.8 or later is recommended. Parsing of html labels may not work properly.')
            html_text = Combine(Literal('<<') + OneOrMore(CharsNotIn(',]')))
        float_number = Combine(Optional(minus) + OneOrMore(Word(nums + '.'))).setName('float_number')
        ID = (alphastring_ | html_text | float_number | quoted_string | identifier).setName('ID')
        righthand_id = (float_number | ID).setName('righthand_id')
        port_angle = (at + ID).setName('port_angle')
        port_location = (OneOrMore(Group(colon + ID)) | Group(colon + lparen + ID + comma + ID + rparen)).setName('port_location')
        port = Combine(Group(port_location + Optional(port_angle)) | Group(port_angle + Optional(port_location))).setName('port')
        node_id = ID + Optional(port)
        a_list = OneOrMore(ID + Optional(equals + righthand_id) + Optional(comma.suppress())).setName('a_list')
        attr_list = OneOrMore(lbrack + Optional(a_list) + rbrack).setName('attr_list').setResultsName('attrlist')
        attr_stmt = ((graph_ | node_ | edge_) + attr_list).setName('attr_stmt')
        edgeop = (Literal('--') | Literal('->')).setName('edgeop')
        stmt_list = Forward()
        graph_stmt = (lbrace + Optional(stmt_list) + rbrace + Optional(semi)).setName('graph_stmt')
        edge_point = Forward()
        edgeRHS = OneOrMore(edgeop + edge_point)
        edge_stmt = edge_point + edgeRHS + Optional(attr_list)
        subgraph = (Optional(subgraph_, '') + Optional(ID, '') + Group(graph_stmt)).setName('subgraph').setResultsName('ssubgraph')
        edge_point <<= subgraph | graph_stmt | node_id
        node_stmt = (node_id + Optional(attr_list) + Optional(semi)).setName('node_stmt')
        assignment = (ID + equals + righthand_id).setName('assignment')
        stmt = (assignment | edge_stmt | attr_stmt | subgraph | graph_stmt | node_stmt).setName('stmt')
        stmt_list <<= OneOrMore(stmt + Optional(semi))
        graphparser = (Optional(strict_, 'notstrict') + (graph_ | digraph_) + Optional(ID, '') + lbrace + Group(Optional(stmt_list)) + rbrace).setResultsName('graph')
        singleLineComment = Group('//' + restOfLine) | Group('#' + restOfLine)
        graphparser.ignore(singleLineComment)
        graphparser.ignore(cStyleComment)
        node_id.setParseAction(self._proc_node_id)
        assignment.setParseAction(self._proc_attr_assignment)
        a_list.setParseAction(self._proc_attr_list)
        edge_stmt.setParseAction(self._proc_edge_stmt)
        node_stmt.setParseAction(self._proc_node_stmt)
        attr_stmt.setParseAction(self._proc_default_attr_stmt)
        attr_list.setParseAction(self._proc_attr_list_combine)
        subgraph.setParseAction(self._proc_subgraph_stmt)
        graphparser.setParseAction(self._main_graph_stmt)
        return graphparser

    def build_graph(self, graph, tokens):
        subgraph = None
        for element in tokens:
            cmd = element[0]
            if cmd == ADD_NODE:
                cmd, nodename, opts = element
                node = graph.add_node(nodename, **opts)
                graph.allitems.append(node)
            elif cmd == ADD_EDGE:
                cmd, src, dest, opts = element
                srcport = destport = ''
                if isinstance(src, tuple):
                    srcport = src[1]
                    src = src[0]
                if isinstance(dest, tuple):
                    destport = dest[1]
                    dest = dest[0]
                edge = graph.add_edge(src, dest, srcport, destport, **opts)
                graph.allitems.append(edge)
            elif cmd in [ADD_GRAPH_TO_NODE_EDGE, ADD_GRAPH_TO_GRAPH_EDGE, ADD_NODE_TO_GRAPH_EDGE]:
                cmd, src, dest, opts = element
                srcport = destport = ''
                if isinstance(src, tuple):
                    srcport = src[1]
                if isinstance(dest, tuple):
                    destport = dest[1]
                if not cmd == ADD_NODE_TO_GRAPH_EDGE:
                    if cmd == ADD_GRAPH_TO_NODE_EDGE:
                        src = subgraph
                    else:
                        src = prev_subgraph
                        dest = subgraph
                else:
                    dest = subgraph
                edges = graph.add_special_edge(src, dest, srcport, destport, **opts)
                graph.allitems.extend(edges)
            elif cmd == SET_GRAPH_ATTR:
                graph.set_attr(**element[1])
            elif cmd == SET_DEF_NODE_ATTR:
                graph.add_default_node_attr(**element[1])
                defattr = DotDefaultAttr('node', **element[1])
                graph.allitems.append(defattr)
            elif cmd == SET_DEF_EDGE_ATTR:
                graph.add_default_edge_attr(**element[1])
                defattr = DotDefaultAttr('edge', **element[1])
                graph.allitems.append(defattr)
            elif cmd == SET_DEF_GRAPH_ATTR:
                graph.add_default_graph_attr(**element[1])
                defattr = DotDefaultAttr('graph', **element[1])
                graph.allitems.append(defattr)
                graph.attr.update(**element[1])
            elif cmd == ADD_SUBGRAPH:
                cmd, name, elements = element
                if subgraph:
                    prev_subgraph = subgraph
                subgraph = graph.add_subgraph(name)
                subgraph = self.build_graph(subgraph, elements)
                graph.allitems.append(subgraph)
        return graph

    def build_top_graph(self, tokens):
        """Build a DotGraph instance from parsed data"""
        strict = tokens[0] == 'strict'
        graphtype = tokens[1]
        directed = graphtype == 'digraph'
        graphname = tokens[2]
        graph = DotGraph(graphname, strict, directed)
        self.graph = self.build_graph(graph, tokens[3])

    def parse_dot_data(self, data):
        """Parse dot data and return a DotGraph instance"""
        try:
            try:
                self.dotparser.parseWithTabs()
            except:
                log.warning('Old version of pyparsing. Parser may not work correctly')
            if os.sys.version_info[0] >= 3 and isinstance(data, bytes):
                data = data.decode()
            ndata = data.replace('\\\n', '')
            tokens = self.dotparser.parseString(ndata)
            self.build_top_graph(tokens[0])
            return self.graph
        except ParseException as err:
            raise

    def parse_dot_data_debug(self, data):
        """Parse dot data"""
        try:
            try:
                self.dotparser.parseWithTabs()
            except:
                log.warning('Old version of pyparsing. Parser may not work correctly')
            tokens = self.dotparser.parseString(data)
            self.build_top_graph(tokens[0])
            return tokens[0]
        except ParseException as err:
            print(err.line)
            print(' ' * (err.column - 1) + '^')
            print(err)
            return None