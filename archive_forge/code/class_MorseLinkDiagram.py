from ..sage_helper import _within_sage
from ..graphs import CyclicList, Digraph
from .links import CrossingStrand, Crossing, Strand, Link
from .orthogonal import basic_topological_numbering
from .tangles import join_strands, RationalTangle
class MorseLinkDiagram:
    """
    A planar link diagram with a height function on R^2 which
    is Morse on the link.
    """

    def __init__(self, link, solver='GLPK'):
        self.link = link = link.copy()
        morse, values = morse_via_LP(link, solver)
        self.morse_number = morse
        self.bends = set(have_positive_value(values[3]))
        self.faces = link.faces()
        self.exterior = self.faces[next(iter(have_positive_value(values[4])))]
        for c in have_positive_value(values[0]):
            c.kind = 'horizontal'
        for c in have_positive_value(values[1]):
            c.kind = 'vertical'
        self.orient_edges()
        self.set_heights()
        self.upward_snakes()

    def orient_edges(self):
        """
        Orients the edges of the link (that is, its CrossingStrands) with
        respect to the height function.

        sage: L = Link('K3a1')
        sage: D = MorseLinkDiagram(L)
        sage: orients = D.orientations.values()
        sage: len(orients) == 4*len(L.crossings)
        True
        sage: sorted(orients)
        ['down', 'down', 'max', 'max', 'max', 'max', 'min', 'min', 'min', 'min', 'up', 'up']
        sage: list(orients).count('max') == 2*D.morse_number
        True
        """

        def expand_orientation(cs, kind):
            c, i = cs
            kinds = CyclicList(['up', 'down', 'down', 'up'])
            if c.kind in 'horizontal':
                s = 0 if i in [0, 3] else 2
            elif c.kind == 'vertical':
                s = 1 if i in [2, 3] else 3
            if kind in ['down', 'max']:
                s += 2
            return [(CrossingStrand(c, i), kinds[i + s]) for i in range(4)]
        orientations = ImmutableValueDict()
        cs = list(self.bends)[0]
        co = cs.opposite()
        orientations[cs] = 'max'
        orientations[co] = 'max'
        current = [cs, co]
        while len(current):
            new = []
            for cs in current:
                for cn, kind in expand_orientation(cs, orientations[cs]):
                    co = cn.opposite()
                    if cn in self.bends or co in self.bends:
                        kind = {'up': 'min', 'down': 'max'}[kind]
                    if co not in orientations:
                        new.append(co)
                    orientations[cn] = kind
                    orientations[co] = {'up': 'down', 'down': 'up', 'max': 'max', 'min': 'min'}[kind]
            current = new
        self.orientations = orientations

    def strands_below(self, crossing):
        """
        The two upward strands below the crossing.
        """
        kinds = self.orientations
        a = CrossingStrand(crossing, 0)
        b = a.rotate()
        upmin = set(['up', 'min'])
        test_a = kinds[a] in upmin
        while True:
            test_b = kinds[b] in upmin
            if test_a and test_b:
                return (a, b)
            a, b = (b, b.rotate())
            test_a = test_b

    def adjacent_upwards(self, crossing_strand):
        a, b = (crossing_strand.rotate(), crossing_strand.rotate(-1))
        if self.orientations[a] in ['up', 'min']:
            return a
        else:
            assert self.orientations[b] in ['up', 'min']
            return b

    def digraph(self):
        """
        The directed graph whose vertices are the mins/maxes of the height
        function together with the crossings, and where the edges come from
        the link and are directed upwards with respect to the height function.

        sage: L = Link('K4a1')
        sage: D = MorseLinkDiagram(L)
        sage: G = D.digraph()
        sage: len(G.vertices)
        8
        """
        G = Digraph()
        kinds = self.orientations
        for cs in self.bends:
            c, d = (cs.crossing, cs.opposite().crossing)
            if kinds[cs] == 'min':
                (G.add_edge(cs, c), G.add_edge(cs, d))
            elif kinds[cs] == 'max':
                (G.add_edge(c, cs), G.add_edge(d, cs))
        for cs, kind in kinds.items():
            if kind == 'up':
                c, d = (cs.crossing, cs.opposite().crossing)
                G.add_edge(d, c)
        return G

    def set_heights(self):
        """
        Assigns a height to each min/max and crossing of the diagram.
        """
        D = self.digraph()
        self.heights = basic_topological_numbering(D)

    def upward_snakes(self):
        """
        Resolve all the crossings vertically and snip all the mins/maxes. The
        resulting pieces are the UpwardSnakes.  For a diagram in bridge position,
        the number of snakes is just twice the bridge number.

        sage: D = MorseLinkDiagram(Link('8a1'))
        sage: len(D.snakes)
        4
        """
        kinds = self.orientations
        self.snakes = snakes = []
        for cs in self.bends:
            if kinds[cs] == 'min':
                snakes += [UpwardSnake(cs, self), UpwardSnake(cs.opposite(), self)]
        self.strand_to_snake = {}
        for snake in snakes:
            for s in snake:
                self.strand_to_snake[s] = snake
            self.strand_to_snake[snake.final] = snake
        self.pack_snakes()

    def pack_snakes(self):
        """
        Give the snakes horizontal positions.
        """
        snakes, to_snake = (self.snakes, self.strand_to_snake)
        S = Digraph(singles=snakes)
        for c in self.link.crossings:
            a, b = self.strands_below(c)
            S.add_edge(to_snake[a], to_snake[b])
        for b in self.bends:
            a = b.opposite()
            S.add_edge(to_snake[a], to_snake[b])
        snake_pos = basic_topological_numbering(S)
        self.S, self.snake_pos = (S, snake_pos)
        heights = self.heights
        max_height = max(heights.values())
        snakes_at_height = {}
        for h in range(max_height + 1):
            at_this_height = []
            for snake in snakes:
                if heights[snake[0].crossing] <= h <= heights[snake[-1].crossing]:
                    at_this_height.append(snake)
            at_this_height.sort(key=lambda s: snake_pos[s])
            for i, s in enumerate(at_this_height):
                snakes_at_height[s, h] = i
        self.snakes_at_height = snakes_at_height

    def is_bridge(self):
        """
        Returns whether the link is in bridge position with respect to this
        height function.
        """
        return all((i == j for i, j in enumerate(sorted(self.snake_pos.values()))))

    def bridge(self):
        if not self.is_bridge():
            raise ValueError("Morse function doesn't give a bridge diagram")

        def to_index(cs):
            return self.snake_pos[self.strand_to_snake[cs]]
        cross_data = []
        for c in self.link.crossings:
            a, b = self.strands_below(c)
            i, j = (to_index(a), to_index(b))
            assert i < j
            cross = (i, j) if a.strand_index % 2 else (j, i)
            cross_data.append((self.heights[c], cross))
        cross_data.sort()

        def bottom_pairing(snake):
            cs = snake[0]
            return tuple(sorted([to_index(cs), to_index(cs.opposite())]))
        bottom = set((bottom_pairing(snake) for snake in self.snakes))

        def top_pairing(snake):
            cs = snake[-1]
            cn = self.adjacent_upwards(snake.final)
            return tuple(sorted([to_index(cs), to_index(cn)]))
        top = set((top_pairing(snake) for snake in self.snakes))
        return BridgeDiagram(bottom, [cd[1] for cd in cross_data], top)