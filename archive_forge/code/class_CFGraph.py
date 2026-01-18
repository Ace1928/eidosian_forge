import collections
import functools
import sys
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
from numba.core.utils import PYVERSION
class CFGraph(object):
    """
    Generic (almost) implementation of a Control Flow Graph.
    """

    def __init__(self):
        self._nodes = set()
        self._preds = _DictOfContainers(set)
        self._succs = _DictOfContainers(set)
        self._edge_data = {}
        self._entry_point = None

    def add_node(self, node):
        """
        Add *node* to the graph.  This is necessary before adding any
        edges from/to the node.  *node* can be any hashable object.
        """
        self._nodes.add(node)

    def add_edge(self, src, dest, data=None):
        """
        Add an edge from node *src* to node *dest*, with optional
        per-edge *data*.
        If such an edge already exists, it is replaced (duplicate edges
        are not possible).
        """
        if src not in self._nodes:
            raise ValueError('Cannot add edge as src node %s not in nodes %s' % (src, self._nodes))
        if dest not in self._nodes:
            raise ValueError('Cannot add edge as dest node %s not in nodes %s' % (dest, self._nodes))
        self._add_edge(src, dest, data)

    def successors(self, src):
        """
        Yield (node, data) pairs representing the successors of node *src*.
        (*data* will be None if no data was specified when adding the edge)
        """
        for dest in self._succs[src]:
            yield (dest, self._edge_data[src, dest])

    def predecessors(self, dest):
        """
        Yield (node, data) pairs representing the predecessors of node *dest*.
        (*data* will be None if no data was specified when adding the edge)
        """
        for src in self._preds[dest]:
            yield (src, self._edge_data[src, dest])

    def set_entry_point(self, node):
        """
        Set the entry point of the graph to *node*.
        """
        assert node in self._nodes
        self._entry_point = node

    def process(self):
        """
        Compute essential properties of the control flow graph.  The graph
        must have been fully populated, and its entry point specified. Other
        graph properties are computed on-demand.
        """
        if self._entry_point is None:
            raise RuntimeError('no entry point defined!')
        self._eliminate_dead_blocks()

    def dominators(self):
        """
        Return a dictionary of {node -> set(nodes)} mapping each node to
        the nodes dominating it.

        A node D dominates a node N when any path leading to N must go through D
        """
        return self._doms

    def post_dominators(self):
        """
        Return a dictionary of {node -> set(nodes)} mapping each node to
        the nodes post-dominating it.

        A node P post-dominates a node N when any path starting from N must go
        through P.
        """
        return self._post_doms

    def immediate_dominators(self):
        """
        Return a dictionary of {node -> node} mapping each node to its
        immediate dominator (idom).

        The idom(B) is the closest strict dominator of V
        """
        return self._idom

    def dominance_frontier(self):
        """
        Return a dictionary of {node -> set(nodes)} mapping each node to
        the nodes in its dominance frontier.

        The dominance frontier _df(N) is the set of all nodes that are
        immediate successors to blocks dominated by N but which aren't
        strictly dominated by N
        """
        return self._df

    def dominator_tree(self):
        """
        return a dictionary of {node -> set(nodes)} mapping each node to
        the set of nodes it immediately dominates

        The domtree(B) is the closest strict set of nodes that B dominates
        """
        return self._domtree

    @functools.cached_property
    def _exit_points(self):
        return self._find_exit_points()

    @functools.cached_property
    def _doms(self):
        return self._find_dominators()

    @functools.cached_property
    def _back_edges(self):
        return self._find_back_edges()

    @functools.cached_property
    def _topo_order(self):
        return self._find_topo_order()

    @functools.cached_property
    def _descs(self):
        return self._find_descendents()

    @functools.cached_property
    def _loops(self):
        return self._find_loops()

    @functools.cached_property
    def _in_loops(self):
        return self._find_in_loops()

    @functools.cached_property
    def _post_doms(self):
        return self._find_post_dominators()

    @functools.cached_property
    def _idom(self):
        return self._find_immediate_dominators()

    @functools.cached_property
    def _df(self):
        return self._find_dominance_frontier()

    @functools.cached_property
    def _domtree(self):
        return self._find_dominator_tree()

    def descendents(self, node):
        """
        Return the set of descendents of the given *node*, in topological
        order (ignoring back edges).
        """
        return self._descs[node]

    def entry_point(self):
        """
        Return the entry point node.
        """
        assert self._entry_point is not None
        return self._entry_point

    def exit_points(self):
        """
        Return the computed set of exit nodes (may be empty).
        """
        return self._exit_points

    def backbone(self):
        """
        Return the set of nodes constituting the graph's backbone.
        (i.e. the nodes that every path starting from the entry point
         must go through).  By construction, it is non-empty: it contains
         at least the entry point.
        """
        return self._post_doms[self._entry_point]

    def loops(self):
        """
        Return a dictionary of {node -> loop} mapping each loop header
        to the loop (a Loop instance) starting with it.
        """
        return self._loops

    def in_loops(self, node):
        """
        Return the list of Loop objects the *node* belongs to,
        from innermost to outermost.
        """
        return [self._loops[x] for x in self._in_loops.get(node, ())]

    def dead_nodes(self):
        """
        Return the set of dead nodes (eliminated from the graph).
        """
        return self._dead_nodes

    def nodes(self):
        """
        Return the set of live nodes.
        """
        return self._nodes

    def topo_order(self):
        """
        Return the sequence of nodes in topological order (ignoring back
        edges).
        """
        return self._topo_order

    def topo_sort(self, nodes, reverse=False):
        """
        Iterate over the *nodes* in topological order (ignoring back edges).
        The sort isn't guaranteed to be stable.
        """
        nodes = set(nodes)
        it = self._topo_order
        if reverse:
            it = reversed(it)
        for n in it:
            if n in nodes:
                yield n

    def dump(self, file=None):
        """
        Dump extensive debug information.
        """
        import pprint
        file = file or sys.stdout
        if 1:
            print('CFG adjacency lists:', file=file)
            self._dump_adj_lists(file)
        print('CFG dominators:', file=file)
        pprint.pprint(self._doms, stream=file)
        print('CFG post-dominators:', file=file)
        pprint.pprint(self._post_doms, stream=file)
        print('CFG back edges:', sorted(self._back_edges), file=file)
        print('CFG loops:', file=file)
        pprint.pprint(self._loops, stream=file)
        print('CFG node-to-loops:', file=file)
        pprint.pprint(self._in_loops, stream=file)
        print('CFG backbone:', file=file)
        pprint.pprint(self.backbone(), stream=file)

    def render_dot(self, filename='numba_cfg.dot'):
        """Render the controlflow graph with GraphViz DOT via the
        ``graphviz`` python binding.

        Returns
        -------
        g : graphviz.Digraph
            Use `g.view()` to open the graph in the default PDF application.
        """
        try:
            import graphviz as gv
        except ImportError:
            raise ImportError('The feature requires `graphviz` but it is not available. Please install with `pip install graphviz`')
        g = gv.Digraph(filename=filename)
        for n in self._nodes:
            g.node(str(n))
        for n in self._nodes:
            for edge in self._succs[n]:
                g.edge(str(n), str(edge))
        return g

    def _add_edge(self, from_, to, data=None):
        self._preds[to].add(from_)
        self._succs[from_].add(to)
        self._edge_data[from_, to] = data

    def _remove_node_edges(self, node):
        for succ in self._succs.pop(node, ()):
            self._preds[succ].remove(node)
            del self._edge_data[node, succ]
        for pred in self._preds.pop(node, ()):
            self._succs[pred].remove(node)
            del self._edge_data[pred, node]

    def _dfs(self, entries=None):
        if entries is None:
            entries = (self._entry_point,)
        seen = set()
        stack = list(entries)
        while stack:
            node = stack.pop()
            if node not in seen:
                yield node
                seen.add(node)
                for succ in self._succs[node]:
                    stack.append(succ)

    def _eliminate_dead_blocks(self):
        """
        Eliminate all blocks not reachable from the entry point, and
        stash them into self._dead_nodes.
        """
        live = set()
        for node in self._dfs():
            live.add(node)
        self._dead_nodes = self._nodes - live
        self._nodes = live
        for dead in self._dead_nodes:
            self._remove_node_edges(dead)

    def _find_exit_points(self):
        """
        Compute the graph's exit points.
        """
        exit_points = set()
        for n in self._nodes:
            if not self._succs.get(n):
                exit_points.add(n)
        return exit_points

    def _find_postorder(self):
        succs = self._succs
        back_edges = self._back_edges
        post_order = []
        seen = set()
        post_order = []

        def dfs_rec(node):
            if node not in seen:
                seen.add(node)
                stack.append((post_order.append, node))
                for dest in succs[node]:
                    if (node, dest) not in back_edges:
                        stack.append((dfs_rec, dest))
        stack = [(dfs_rec, self._entry_point)]
        while stack:
            cb, data = stack.pop()
            cb(data)
        return post_order

    def _find_immediate_dominators(self):

        def intersect(u, v):
            while u != v:
                while idx[u] < idx[v]:
                    u = idom[u]
                while idx[u] > idx[v]:
                    v = idom[v]
            return u
        entry = self._entry_point
        preds_table = self._preds
        order = self._find_postorder()
        idx = {e: i for i, e in enumerate(order)}
        idom = {entry: entry}
        order.pop()
        order.reverse()
        changed = True
        while changed:
            changed = False
            for u in order:
                new_idom = functools.reduce(intersect, (v for v in preds_table[u] if v in idom))
                if u not in idom or idom[u] != new_idom:
                    idom[u] = new_idom
                    changed = True
        return idom

    def _find_dominator_tree(self):
        idom = self._idom
        domtree = _DictOfContainers(set)
        for u, v in idom.items():
            if u not in domtree:
                domtree[u] = set()
            if u != v:
                domtree[v].add(u)
        return domtree

    def _find_dominance_frontier(self):
        idom = self._idom
        preds_table = self._preds
        df = {u: set() for u in idom}
        for u in idom:
            if len(preds_table[u]) < 2:
                continue
            for v in preds_table[u]:
                while v != idom[u]:
                    df[v].add(u)
                    v = idom[v]
        return df

    def _find_dominators_internal(self, post=False):
        if post:
            entries = set(self._exit_points)
            preds_table = self._succs
            succs_table = self._preds
        else:
            entries = set([self._entry_point])
            preds_table = self._preds
            succs_table = self._succs
        if not entries:
            raise RuntimeError('no entry points: dominator algorithm cannot be seeded')
        doms = {}
        for e in entries:
            doms[e] = set([e])
        todo = []
        for n in self._nodes:
            if n not in entries:
                doms[n] = set(self._nodes)
                todo.append(n)
        while todo:
            n = todo.pop()
            if n in entries:
                continue
            new_doms = set([n])
            preds = preds_table[n]
            if preds:
                new_doms |= functools.reduce(set.intersection, [doms[p] for p in preds])
            if new_doms != doms[n]:
                assert len(new_doms) < len(doms[n])
                doms[n] = new_doms
                todo.extend(succs_table[n])
        return doms

    def _find_dominators(self):
        return self._find_dominators_internal(post=False)

    def _find_post_dominators(self):
        dummy_exit = object()
        self._exit_points.add(dummy_exit)
        for loop in self._loops.values():
            if not loop.exits:
                for b in loop.body:
                    self._add_edge(b, dummy_exit)
        pdoms = self._find_dominators_internal(post=True)
        del pdoms[dummy_exit]
        for doms in pdoms.values():
            doms.discard(dummy_exit)
        self._remove_node_edges(dummy_exit)
        self._exit_points.remove(dummy_exit)
        return pdoms

    def _find_back_edges(self, stats=None):
        """
        Find back edges.  An edge (src, dest) is a back edge if and
        only if *dest* dominates *src*.
        """
        if stats is not None:
            if not isinstance(stats, dict):
                raise TypeError(f'*stats* must be a dict; got {type(stats)}')
            stats.setdefault('iteration_count', 0)
        back_edges = set()
        stack = []
        succs_state = {}
        entry_point = self.entry_point()
        checked = set()

        def push_state(node):
            stack.append(node)
            succs_state[node] = [dest for dest in self._succs[node]]
        push_state(entry_point)
        iter_ct = 0
        while stack:
            iter_ct += 1
            tos = stack[-1]
            tos_succs = succs_state[tos]
            if tos_succs:
                cur_node = tos_succs.pop()
                if cur_node in stack:
                    back_edges.add((tos, cur_node))
                elif cur_node not in checked:
                    push_state(cur_node)
            else:
                stack.pop()
                checked.add(tos)
        if stats is not None:
            stats['iteration_count'] += iter_ct
        return back_edges

    def _find_topo_order(self):
        succs = self._succs
        back_edges = self._back_edges
        post_order = []
        seen = set()

        def _dfs_rec(node):
            if node not in seen:
                seen.add(node)
                for dest in succs[node]:
                    if (node, dest) not in back_edges:
                        _dfs_rec(dest)
                post_order.append(node)
        _dfs_rec(self._entry_point)
        post_order.reverse()
        return post_order

    def _find_descendents(self):
        descs = {}
        for node in reversed(self._topo_order):
            descs[node] = node_descs = set()
            for succ in self._succs[node]:
                if (node, succ) not in self._back_edges:
                    node_descs.add(succ)
                    node_descs.update(descs[succ])
        return descs

    def _find_loops(self):
        """
        Find the loops defined by the graph's back edges.
        """
        bodies = {}
        for src, dest in self._back_edges:
            header = dest
            body = set([header])
            queue = [src]
            while queue:
                n = queue.pop()
                if n not in body:
                    body.add(n)
                    queue.extend(self._preds[n])
            if header in bodies:
                bodies[header].update(body)
            else:
                bodies[header] = body
        loops = {}
        for header, body in bodies.items():
            entries = set()
            exits = set()
            for n in body:
                entries.update(self._preds[n] - body)
                exits.update(self._succs[n] - body)
            loop = Loop(header=header, body=body, entries=entries, exits=exits)
            loops[header] = loop
        return loops

    def _find_in_loops(self):
        loops = self._loops
        in_loops = dict(((n, []) for n in self._nodes))
        for loop in sorted(loops.values(), key=lambda loop: len(loop.body)):
            for n in loop.body:
                in_loops[n].append(loop.header)
        return in_loops

    def _dump_adj_lists(self, file):
        adj_lists = dict(((src, sorted(list(dests))) for src, dests in self._succs.items()))
        import pprint
        pprint.pprint(adj_lists, stream=file)

    def __eq__(self, other):
        if not isinstance(other, CFGraph):
            return NotImplemented
        for x in ['_nodes', '_edge_data', '_entry_point', '_preds', '_succs']:
            this = getattr(self, x, None)
            that = getattr(other, x, None)
            if this != that:
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)