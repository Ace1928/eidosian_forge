from io import BytesIO
from . import osutils, progress, trace
from .i18n import gettext
from .ui import ui_factory
def add_edge_hashes(self, lines, tag):
    """Update edge_hashes to include the given lines.

        :param lines: The lines to update the hashes for.
        :param tag: A tag uniquely associated with these lines (i.e. file-id)
        """
    for my_hash in self.iter_edge_hashes(lines):
        self.edge_hashes.setdefault(my_hash, set()).add(tag)