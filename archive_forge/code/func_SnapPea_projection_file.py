import time
from string import ascii_lowercase
from .gui import tkMessageBox
from .vertex import Vertex
from .arrow import Arrow, default_arrow_params
from .crossings import Crossing, ECrossing
from .smooth import TikZPicture
def SnapPea_projection_file(self):
    """
        Returns a string containing the contents of a SnapPea link
        projection file.
        """
    has_virtual_crossings = any((crossing.is_virtual for crossing in self.Crossings))
    result = ''
    result += '% Virtual Link Projection\n' if has_virtual_crossings else '% Link Projection\n'
    components = self.arrow_components()
    result += '%d\n' % len(components)
    for component in components:
        first = self.Vertices.index(component[0].start)
        last = self.Vertices.index(component[-1].end)
        result += '%4.1d %4.1d\n' % (first, last)
    result += '%d\n' % len(self.Vertices)
    for vertex in self.Vertices:
        result += '%5.1d %5.1d\n' % vertex.point()
    result += '%d\n' % len(self.Arrows)
    for arrow in self.Arrows:
        start_index = self.Vertices.index(arrow.start)
        end_index = self.Vertices.index(arrow.end)
        result += '%4.1d %4.1d\n' % (start_index, end_index)
    result += '%d\n' % len(self.Crossings)
    for crossing in self.Crossings:
        under = self.Arrows.index(crossing.under)
        over = self.Arrows.index(crossing.over)
        is_virtual = 'v' if crossing.is_virtual else 'r'
        result += '%4s %4.1d %4.1d\n' % (is_virtual, under, over) if has_virtual_crossings else '%4.1d %4.1d\n' % (under, over)
    if self.ActiveVertex:
        result += '%d\n' % self.Vertices.index(self.ActiveVertex)
    else:
        result += '-1\n'
    return result