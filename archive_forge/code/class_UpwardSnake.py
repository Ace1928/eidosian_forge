from ..sage_helper import _within_sage
from ..graphs import CyclicList, Digraph
from .links import CrossingStrand, Crossing, Strand, Link
from .orthogonal import basic_topological_numbering
from .tangles import join_strands, RationalTangle
class UpwardSnake(tuple):
    """
    Start with an MorseLinkDiagram, resolve all the crossings vertically,
    and snip all the mins/maxes.  The resulting pieces are the UpwardSnakes.
    """

    def __new__(self, crossing_strand, link):
        cs, kind = (crossing_strand, link.orientations)
        assert kind[cs] == 'min'
        snake = [cs]
        while True:
            ca, cb = (cs.rotate(1).opposite(), cs.rotate(-1).opposite())
            if kind[ca] == 'max' or kind[cb] == 'max':
                break
            if kind[ca] == 'up':
                cs = ca
            else:
                assert kind[cb] == 'up'
                cs = cb
            snake.append(cs)
        ans = tuple.__new__(UpwardSnake, snake)
        ans.final = ca if kind[ca] == 'max' else cb
        heights = [link.heights[cs.crossing] for cs in ans]
        assert heights == sorted(heights)
        ans.heights = heights
        return ans