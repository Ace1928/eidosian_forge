from .. import FatGraph, FatEdge, Link, Crossing
from ..links.links import CrossingEntryPoint
from ..links.ordered_set import OrderedSet
from .Base64LikeDT import (decode_base64_like_DT_code, encode_base64_like_DT_code)
class DTFatGraph(FatGraph):
    edge_class = DTFatEdge

    def __init__(self):
        self.vertices = OrderedSet()
        self.edges = OrderedSet()
        self.incidence_dict = {}
        self.Edge = self.__class__.edge_class
        self.marked_valences = {v: 0 for v in self.vertices}
        self.stack = []
        self.pushed = False

    def add_edge(self, x, y):
        edge = FatGraph.add_edge(self, x, y)
        self.marked_valences[x[0]] = 0
        self.marked_valences[y[0]] = 0
        return edge

    def mark(self, edgelist):
        """
        Mark all edges in the list.
        """
        vertices = set()
        for edge in edgelist:
            edge.marked = True
            vertices.update(edge)
        for v in vertices:
            self.marked_valences[v] = self.marked_valence(v)

    def marked_valence(self, vertex):
        """
        Compute the marked valence of a vertex.
        """
        valence = 0
        for e in self.incidence_dict[vertex]:
            if e.marked:
                valence += 1
        return valence

    def clear(self):
        """
        Remove all edge markings.
        """
        for e in self.edges:
            e.marked = False
        self.marked_valences = {v: 0 for v in self.vertices}

    def push(self, flips):
        """
        Save the state of this DTFatGraph before doing the flips.
        The flips will be done by the pop.
        """
        if not self.pushed:
            self.pushed = True
            return
        self.stack.append([flips, [((e[0], e.slots[0]), (e[1], e.slots[1]), e.component) for e in self.edges if e.marked], [((e[0], e.slots[0]), (e[1], e.slots[1]), e.component) for e in self.edges if not e.marked]])

    def pop(self):
        """
        Restore the state of this DTFatGraph and perform the saved flips.
        """
        self.edges = set()
        flips, marked, unmarked = self.stack.pop()
        for x, y, component in marked:
            edge = self.add_edge(x, y)
            edge.component = component
            edge.marked = True
        for x, y, component in unmarked:
            edge = self.add_edge(x, y)
            edge.component = component
        return flips

    def clear_stack(self):
        """
        Reset the state stack.
        """
        self.stack = []
        self.pushed = False

    def path(self, vertex, edge):
        """
        Return an iterator which iterates through the edges of a
        component, starting at the given edge, in the direction
        determined by the vertex.
        """
        if vertex not in edge:
            raise ValueError('That vertex is not an endpoint of the edge.')
        forward = bool(vertex == edge[0])
        return DTPath(edge, self, forward)

    def marked_arc(self, vertex):
        """
        Given a vertex with marked valence 2, find the maximal marked
        arc containing the vertex for which all interior edges have
        marked valence 2.  If the marked subgraph is a circle, or a
        dead end is reached, raise :class:`ValueError`.  Return a list of
        edges in the arc.
        """
        left_path, right_path, vertices = ([], [], set())
        vertices.add(vertex)
        try:
            left, right = [e for e in self(vertex) if e.marked]
        except ValueError:
            raise RuntimeError('Vertex must have two marked edges.')
        for edge, path in ((left, left_path), (right, right_path)):
            V = vertex
            while True:
                path.append(edge)
                V = edge(V)
                if V == vertex:
                    raise ValueError('Marked graph is a circle')
                edges = [e for e in self(V) if e.marked and e != edge]
                if len(edges) == 0:
                    raise ValueError('Marked graph has a dead end at %s.' % V)
                if len(edges) > 1:
                    break
                else:
                    vertices.add(V)
                    edge = edges.pop()
        left_path.reverse()
        return left_path + right_path

    def unmarked_arc(self, vertex):
        """
        Starting at this vertex, find an unmarked edge and follow its
        component until we run into a vertex with at least one marked
        edge.  Remove loops to get an embedded arc. Return the list
        of edges traversed by the embedded arc.
        """
        valence = self.marked_valence(vertex)
        if valence == 4:
            raise ValueError('Vertex must have unmarked edges.')
        if valence == 0:
            raise ValueError('Vertex must be in the marked subgraph.')
        edges, vertices, seen = ([], [], set())
        for first_edge in self(vertex):
            if not first_edge.marked:
                break
        for edge in self.path(vertex, first_edge):
            edges.append(edge)
            vertex = edge(vertex)
            if self.marked_valence(vertex) > 0:
                break
            if vertex in seen:
                n = vertices.index(vertex)
                edges = edges[:n + 1]
                vertices = vertices[:n + 1]
                seen = set(vertices)
            else:
                vertices.append(vertex)
                seen.add(vertex)
        return (edges, vertex)

    def bridge(self, marked_arc):
        """
        Try to find an embedded path of unmarked edges joining a
        vertex in the given arc to a vertex of the marked subgraph
        which lies in the complement of the interior of the arc.  This
        uses a depth-first search, and raises :class:`ValueError` on failure.
        Returns a triple (first vertex, last vertex, edge path).

        Suppose the marked subgraph has no vertices with marked
        valence 3 and has a unique planar embedding.  Choose a maximal
        arc with all interior vertices having marked valence 2. Then
        adding a bridge from that arc produces a new graph with a
        unique planar embedding.

        In the case where the diagram is prime, a bridge will exist
        for *some* vertex on the arc.  But, e.g. in the case where the
        arc snakes through a twist region, it will only exist for the
        two extremal vertices in the arc.
        """
        e0, e1 = marked_arc[:2]
        v = e0[0] if e0[1] in e1 else e0[1]
        vertex_list = []
        for edge in marked_arc[:-1]:
            v = edge(v)
            vertex_list.append(v)
        for start_vertex in vertex_list:
            vertex = start_vertex
            vertex_set = set(vertex_list)
            edge_path, vertex_path, seen_edges = ([], [], set())
            while True:
                edges = [e for e in self(vertex) if not e.marked and e not in seen_edges and (e(vertex) not in vertex_set)]
                try:
                    new_edge = edges.pop()
                    vertex = new_edge(vertex)
                    edge_path.append(new_edge)
                    if self.marked_valence(vertex) > 0:
                        return (start_vertex, vertex, edge_path)
                    seen_edges.add(new_edge)
                    vertex_path.append(vertex)
                    vertex_set.add(vertex)
                except IndexError:
                    if len(edge_path) == 0:
                        break
                    edge_path.pop()
                    vertex_set.remove(vertex_path.pop())
                    try:
                        vertex = vertex_path[-1]
                    except IndexError:
                        vertex = start_vertex
        raise ValueError('Could not find a bridge.')

    def _boundary_slots(self, edge, side):
        """
        Assume that the marked subFatGraph has been embedded in the
        plane.  This generator starts at a marked FatEdge and walks
        around one of its adjacent boundary curves (left=-1, right=1),
        yielding all of the pairs (v, s) where s is a slot of the
        vertex v which lies on the specified boundary curve, or
        (v, None) if none of the slots at v lie on the curve.  (To
        extend the embedding over an unmarked arc, the ending slots of
        both ends of the arc must lie on the same boundary curve.
        Flipping may be needed to arrange this.)
        """
        if not edge.marked:
            raise ValueError('Must begin at a marked edge.')
        first_vertex = vertex = edge[1]
        while True:
            end = 0 if edge[0] is vertex else 1
            slot = edge.slots[end]
            for k in range(3):
                slot += side
                interior_edge = self(vertex)[slot]
                if not interior_edge.marked:
                    yield (vertex, slot % 4)
                else:
                    break
            if k == 0:
                yield (vertex, None)
            if edge is interior_edge:
                raise ValueError('Marked subgraph has a dead end.')
            edge = interior_edge
            vertex = edge(vertex)
            if vertex is first_vertex:
                break

    def left_slots(self, edge):
        """
        Return the (vertex, slot) pairs on the left boundary curve.
        """
        return set(self._boundary_slots(edge, side=-1))

    def right_slots(self, edge):
        """
        Return the (vertex, slot) pairs on the right boundary curve.
        """
        return set(self._boundary_slots(edge, side=1))

    def flip(self, vertex, force=False):
        """
        Move the edge at the North slot to the South slot, and
        move the edge in the South  slot to the North slot.
        """
        if not force and self.marked_valences[vertex] > 2:
            msg = 'Cannot flip %s with marked valence %d.' % (vertex, self.marked_valences[vertex])
            raise FlippingError(msg)
        self.reorder(vertex, (North, East, South, West))

    def incoming_under(self, vertex):
        first, second, even_over = vertex
        incoming = [e.PD_index() for e in self(vertex) if e[1] is vertex]
        incoming.sort(key=lambda x: x % 2)
        return incoming[0] if even_over else incoming[1]

    def PD_tuple(self, vertex):
        """
        Return the PD labels of the incident edges in order, starting
        with the incoming undercrossing as required for PD codes.
        """
        edgelist = [e.PD_index() for e in self(vertex)]
        n = edgelist.index(self.incoming_under(vertex))
        return tuple(edgelist[n:] + edgelist[:n])

    def flipped(self, vertex):
        """
        Has this vertex been flipped?
        """
        return bool(len([e for e in self(vertex) if e[1] is vertex and e.slots[1] in (2, 3)]) % 2)

    def sign(self, vertex):
        """
        The sign of the crossing corresponding to this vertex.
        See the documentation for Spherogram.link.
        """
        flipped = self.flipped(vertex)
        even_first = bool(vertex[0] % 2 == 0)
        return -1 if flipped ^ vertex[2] ^ even_first else 1

    def KLP_strand(self, vertex, edge):
        """
        Return the SnapPea KLP strand name for the given edge at the
        end opposite to the vertex.
        """
        W = edge(vertex)
        slot = edge.slot(W)
        return 'X' if (slot == 0 or slot == 2) ^ self.flipped(W) else 'Y'

    def KLP_dict(self, vertex, indices):
        """
        Return a dict describing this vertex and its neighbors
        in KLP terminology.

        The translation from our convention is as follows::

                    Y                    Y
                    3                    0
                    ^                    ^
                    |                    |
             0 -----+----> 2 X     1 ----+---> 3 X
                    |                    |
                    |                    |
                    1                    2
               not flipped           flipped

        The indices argument is a dict that assigns an integer
        index to each vertex of the graph.
        """
        KLP = {}
        flipped = self.flipped(vertex)
        edges = self(vertex)
        neighbors = self[vertex]
        strands = [self.KLP_strand(vertex, edge) for edge in edges]
        ids = [indices[v] for v in neighbors]
        KLP['sign'] = 'R' if self.sign(vertex) == 1 else 'L'
        slot = 1 if flipped else 0
        KLP['Xbackward_neighbor'] = ids[slot]
        KLP['Xbackward_strand'] = strands[slot]
        slot = 3 if flipped else 2
        KLP['Xforward_neighbor'] = ids[slot]
        KLP['Xforward_strand'] = strands[slot]
        KLP['Xcomponent'] = edges[slot].component
        slot = 2 if flipped else 1
        KLP['Ybackward_neighbor'] = ids[slot]
        KLP['Ybackward_strand'] = strands[slot]
        slot = 0 if flipped else 3
        KLP['Yforward_neighbor'] = ids[slot]
        KLP['Yforward_strand'] = strands[slot]
        KLP['Ycomponent'] = edges[slot].component
        return KLP