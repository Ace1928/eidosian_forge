import pickle
from .links import Crossing, Strand, Link
from . import planar_isotopy
def _fuse_strands(self, preserve_boundary=False, preserve_components=False):
    """Fuse all strands and delete them, even ones incident to only the boundary (unless
        ``preserve_boundary`` is True). This will eliminate Strands that are loops as well.

        If ``preserve_components`` is True, then do not fuse strands that have the
        ``component_idx`` attribute."""
    for s in reversed(self.crossings):
        if isinstance(s, Strand):
            if preserve_boundary and all((a[0] == self for a in s.adjacent)):
                continue
            if preserve_components and s.component_idx is not None:
                continue
            s.fuse()
            self.crossings.remove(s)