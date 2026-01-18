from .. import FatGraph, FatEdge, Link, Crossing
from ..links.links import CrossingEntryPoint
from ..links.ordered_set import OrderedSet
from .Base64LikeDT import (decode_base64_like_DT_code, encode_base64_like_DT_code)
class DTcodec:
    """
    Codec for DT codes of a link projection.
    """

    def __init__(self, input, flips=None):
        if isinstance(input, (bytes, str, list)):
            self.decode(input, flips)

    def __getitem__(self, n):
        """
        A DTcodec can look up a vertex by either of its labels.
        """
        return self.lookup[n]

    def decode(self, dt, flips=None):
        """
        Accepts input of the following types:
        1) a dt code in either numeric or alphabetical form and a sequence
        of boolean values, one for each successive label in the DT code,
        indicating whether the crossing with that label needs to be
        flipped.  Alphabetical DT codes can also specify signs by appending
        a period followed by a sequence of 0's and 1's.
        2) a DT code with flips set to None.  In this case the flips are
        computed.
        3) a bytes object containing a compact signed DT code.  The
        signed DT code may be provided as a byte string, for which
        the last byte has bit 7 set, as produced by the signed_DT method.
        Alternatives, it can be hex encoded as a string
        beginning with '0x', as produced by the hex_signed_DT method.

        This method constructs a planar FatGraph from its input data.
        """
        if flips is not None:
            flips = [bool(flip) for flip in flips]
        self.flips = flips
        if isinstance(dt, (str,)):
            if dt[:2] == '0x':
                dt_bytes = [int(dt[n:n + 2], 16) for n in range(2, len(dt), 2)]
                self.code, self.flips = self.unpack_signed_DT(dt_bytes)
            elif ord(dt[-1]) & 1 << 7:
                dt_bytes = bytearray(dt)
                self.code, self.flips = self.unpack_signed_DT(dt)
            elif dt[0] in '123456789':
                self.code, self.flips = decode_base64_like_DT_code(dt)
            else:
                parts = dt.split('.')
                self.code = self.convert_alpha(parts[0])
                if len(parts) > 1:
                    self.flips = [d != '0' for d in parts[1]]
        elif isinstance(dt, bytes):
            self.code, self.flips = self.unpack_signed_DT(dt)
        else:
            self.code = dt
        code = self.code
        overcrossings = (sign(x) for comp in code for x in comp)
        evens = [abs(x) for comp in code for x in comp]
        self.size = size = 2 * len(evens)
        pairs = zip(range(1, size, 2), evens)
        self.lookup = lookup = [None for n in range(1 + size)]
        for pair, overcrossing in zip(pairs, overcrossings):
            V = DTvertex(pair, overcrossing)
            m, n = pair
            lookup[m] = lookup[n] = V
        self.fat_graph = G = DTFatGraph()
        N = start = 1
        last_odd = -1
        for c, component in enumerate(code):
            if len(component) == 0:
                continue
            last_odd += 2 * len(component)
            V = self[N]
            while N <= last_odd:
                W = self[N + 1]
                edge = G.add_edge((V, V.exit_slot(N)), (W, W.entry_slot(N + 1)))
                edge.component = c
                N += 1
                V = W
            S = self[start]
            edge = G.add_edge((V, V.exit_slot(N)), (S, S.entry_slot(start)))
            edge.component = c
            start = N = N + 1
        labels = [abs(N) for component in code for N in component]
        if self.flips is None:
            self.embed()
            if G.sign(self[1]) != 1:
                for label in labels:
                    G.flip(self[label], force=True)
            self.flips = [G.flipped(self[label]) for label in labels]
        else:
            for label, flip in zip(labels, self.flips):
                if flip:
                    G.flip(self[label])

    def unpack_signed_DT(self, signed_dt):
        dt = []
        component = []
        flips = []
        for byte in bytearray(signed_dt):
            flips.append(bool(byte & 1 << 6))
            label = (1 + byte & 31) << 1
            if byte & 1 << 5:
                label = -label
            component.append(label)
            if byte & 1 << 7:
                dt.append(tuple(component))
                component = []
        return (dt, flips)

    def convert_alpha(self, code):
        code = string_to_ints(code)
        num_crossings, components = code[:2]
        comp_lengths = code[2:2 + components]
        crossings = [x << 1 for x in code[2 + components:]]
        assert len(crossings) == num_crossings
        return partition_list(crossings, comp_lengths)

    def encode(self, header=True, alphabetical=True, flips=True):
        """
        Returns a string describing the DT code.  Options control
        whether to include the 'DT:' header, whether to use the
        numerical or alphabetical format, and whether to use the
        extended form, which adds flip information for each crossing.
        If flips is set to "auto", only include flips in large links
        (>26 crossings).

        >>> d = DTcodec([(-6,-8,-2,-4)])
        >>> A = d.encode()
        >>> A in ['DT:dadCDAB.0110', 'DT:dadCDAB.1001']
        True
        >>> N = d.encode(alphabetical=False)
        >>> N in ['DT:[(-6,-8,-2,-4)], [0,1,1,0]',
        ...  'DT:[(-6,-8,-2,-4)], [1,0,0,1]']
        True
        >>> d.encode(flips=False)
        'DT:dadCDAB'
        >>> d.encode(alphabetical=False, flips=False)
        'DT:[(-6,-8,-2,-4)]'
        """
        code = self.code
        result = 'DT:' if header else ''
        chunks = [len(component) for component in code]
        num_crossings = sum(chunks)
        is_large = num_crossings > 26
        if flips == 'auto':
            flips = is_large
        if alphabetical:
            if is_large:
                if flips:
                    result += encode_base64_like_DT_code(code, self.flips)
                else:
                    result += encode_base64_like_DT_code(code)
            else:
                prefix_ints = [num_crossings, len(code)]
                prefix_ints += chunks
                code_ints = [x for component in code for x in component]
                alphacode = ''.join((DT_alphabet[n >> 1] for n in code_ints))
                prefix = ''.join((DT_alphabet[n] for n in prefix_ints))
                if flips:
                    alphacode += '.' + ''.join((str(int(f)) for f in self.flips))
                result += prefix + alphacode
        else:
            result += str(code)
            if flips:
                result += ',  %s' % [int(f) for f in self.flips]
            result = result.replace(', ', ',')
        return result

    def embed(self, edge=None):
        """
        Try to flip crossings in the FatGraph until it becomes planar.
        """
        G = self.fat_graph
        if edge is None:
            for edge in G.edges:
                break
        vertices, circle_edges = self.find_circle(edge)
        G.mark(circle_edges)
        first, last, arc_edges = G.bridge(circle_edges[:2])
        self.do_flips(first, arc_edges[0], last, arc_edges[-1])
        G.mark(arc_edges)
        while True:
            try:
                if not self.embed_arc():
                    break
            except EmbeddingError:
                flips = G.pop()
                for vertex in flips:
                    G.flip(vertex)
        self.fat_graph.clear_stack()

    def find_circle(self, first_edge):
        """
        Follow a component, starting at the given (directed) edge,
        until the first time it crosses itself.  Throw away the tail
        to get a cycle.  Return the list of vertices and the list of
        edges traversed by the cycle.
        """
        edges = []
        vertices = [first_edge[0]]
        seen = set(vertices)
        for edge in self.fat_graph.path(first_edge[0], first_edge):
            vertex = edge[1]
            if vertex in seen:
                edges.append(edge)
                break
            else:
                seen.add(vertex)
                vertices.append(vertex)
                edges.append(edge)
        n = vertices.index(vertex)
        edges = edges[n:]
        vertices = vertices[n:]
        return (vertices, edges)

    def get_incomplete_vertex(self):
        """
        Return a vertex with some marked and some unmarked edges.
        If there are any, return a vertex with marked valence 3.
        """
        G = self.fat_graph
        vertices = [v for v in G.vertices if 0 < G.marked_valences[v] < 4]
        vertices.sort(key=lambda v: G.marked_valences[v])
        try:
            return vertices.pop()
        except IndexError:
            return None

    def do_flips(self, v, v_edge, w, w_edge):
        """
        Decide whether v and/or w needs to be flipped in order to add
        an arc from v to w starting with the v_edge and ending with
        the w_edge.  If flips are needed, make them.  If the embedding
        cannot be extended raise an exception.
        """
        G = self.fat_graph
        vslot = G(v).index(v_edge)
        wslot = G(w).index(w_edge)
        for k in range(1, 3):
            ccw_edge = G(v)[vslot + k]
            if ccw_edge.marked:
                break
        if not ccw_edge.marked:
            raise ValueError('Invalid marking')
        left_slots = G.left_slots(ccw_edge)
        right_slots = G.right_slots(ccw_edge)
        v_valence, w_valence = (G.marked_valences[v], G.marked_valences[w])
        if (v, vslot) in left_slots:
            v_slot_side, v_other_side = (left_slots, right_slots)
        else:
            v_slot_side, v_other_side = (right_slots, left_slots)
        w_on_slot_side = w in [x[0] for x in v_slot_side]
        w_on_other_side = w in [x[0] for x in v_other_side]
        if not w_on_slot_side and (not w_on_other_side):
            raise EmbeddingError('Embedding does not extend.')
        if (w, wslot) in v_slot_side:
            if v_valence == w_valence == 2:
                G.push([v, w])
            return
        if w_valence != 2:
            G.flip(v)
            return
        elif v_valence != 2:
            G.flip(w)
            return
        if w_on_slot_side and (not w_on_other_side):
            G.flip(w)
            return
        if w_on_slot_side and w_on_other_side:
            G.push([w])
        G.flip(v)
        if not (w, wslot) in v_other_side:
            G.flip(w)

    def embed_arc(self):
        G = self.fat_graph
        v = self.get_incomplete_vertex()
        if v is None:
            return False
        if G.marked_valences[v] == 2:
            try:
                first, last, arc_edges = G.bridge(G.marked_arc(v))
            except ValueError:
                arc_edges, last = G.unmarked_arc(v)
                first = v
            self.do_flips(first, arc_edges[0], last, arc_edges[-1])
        else:
            arc_edges, last_vertex = G.unmarked_arc(v)
            self.do_flips(last_vertex, arc_edges[-1], v, arc_edges[0])
        G.mark(arc_edges)
        return True

    def signed_DT(self):
        """
        Return a byte sequence containing the signed DT code.

        >>> d = DTcodec([(-6,-8,-2,-4)])
        >>> d2 = DTcodec(d.signed_DT())
        >>> d2.code
        [(-6, -8, -2, -4)]
        """
        code_bytes = bytearray()
        it = iter(self.flips)
        for component in self.code:
            for label in component:
                byte = abs(label)
                byte = (byte >> 1) - 1
                if label < 0:
                    byte |= 1 << 5
                if next(it):
                    byte |= 1 << 6
                code_bytes.append(byte)
            code_bytes[-1] |= 1 << 7
        return bytes(code_bytes)

    def hex_signed_DT(self):
        """
        Return the hex encoding of the signed DT byte sequence.

        >>> d = DTcodec([(-6,-8,-2,-4)])
        >>> d2 = DTcodec(d.hex_signed_DT())
        >>> d2.code
        [(-6, -8, -2, -4)]
        """
        return '0x' + ''.join(('%.2x' % b for b in bytearray(self.signed_DT())))

    def PD_code(self, KnotTheory=False):
        """
        Return a PD code for the projection described by this DT code,
        as a list of lists of 4 integers.  If KnotTheory is set to
        True, return a string that can be used as input to the Knot
        Theory package.

        >>> d = DTcodec([(-6,-8,-2,-4)], [0,1,1,0])
        >>> sorted(d.PD_code())
        [(2, 8, 3, 7), (4, 1, 5, 2), (6, 4, 7, 3), (8, 5, 1, 6)]
        """
        G = self.fat_graph
        PD = [G.PD_tuple(v) for v in G.vertices]
        if KnotTheory:
            return 'PD[%s]' % ', '.join(('X%s' % repr(list(t)) for t in PD))
        return PD

    def link(self):
        G = self.fat_graph
        crossing_dict, slot_dict = ({}, {})
        for v in G.vertices:
            c = Crossing(v[0])
            c.make_tail(0)
            if G.sign(v) == 1:
                c.make_tail(3)
            else:
                c.make_tail(1)
            c.orient()
            crossing_dict[v] = c
            if v.upper_pair() == (0, 2):
                slot_dict[v] = (3, 0, 1, 2)
            else:
                slot_dict[v] = (2, 3, 0, 1) if G.flipped(v) else (0, 1, 2, 3)
        for edge in G.edges:
            v0, v1 = edge
            s0, s1 = edge.slots
            a0, a1 = (slot_dict[v0][s0], slot_dict[v1][s1])
            c0, c1 = (crossing_dict[v0], crossing_dict[v1])
            c0[a0] = c1[a1]
        link = Link(list(crossing_dict.values()), check_planarity=False, build=False)
        assert link.all_crossings_oriented()
        component_starts = []
        i = 1
        for comp in self.code:
            c = self[i]
            if i == c[0]:
                e = slot_dict[c][2] if G.flipped(c) else slot_dict[c][0]
            else:
                e = slot_dict[c][1]
            ce = CrossingEntryPoint(crossing_dict[c], e)
            component_starts.append(ce)
            i += 2 * len(comp)
        link._build_components(component_starts)
        if not link.is_planar():
            raise ValueError('DT code does not seem to define a *planar* diagram')
        return link

    def KLPProjection(self):
        """
        Constructs a python simulation of a SnapPea KLPProjection
        (Kernel Link Projection) structure.  See DTFatGraph.KLP_dict
        and Jeff Weeks' SnapPea file link_projection.h for
        definitions.  Here the KLPCrossings are modeled by
        dictionaries.
        """
        G = self.fat_graph
        vertices = list(G.vertices)
        KLP_indices = {v: n for n, v in enumerate(vertices)}
        KLP_crossings = [G.KLP_dict(v, KLP_indices) for v in vertices]
        return (len(G.vertices), 0, len(self.code), KLP_crossings)

    def exterior(L):
        raise RuntimeError("SnapPy doesn't seem to be available.")