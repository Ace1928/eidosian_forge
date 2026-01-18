import re
from io import BytesIO
from .lazy_import import lazy_import
from fnmatch import fnmatch
from breezy._termcolor import color_string, FG
from breezy import (
from . import controldir, errors, osutils
from . import revision as _mod_revision
from . import trace
from .revisionspec import RevisionSpec, RevisionSpec_revid, RevisionSpec_revno
def _get_writer_plain(self, path, revno, cache_id):
    """Get function for writing uncoloured output"""
    per_line = self._format_perline
    start = self._format_initial % {'path': path, 'revno': revno}
    write = self.outf.write
    if self.cache is not None and cache_id is not None:
        result_list = []
        self.cache[cache_id] = (path, result_list)
        add_to_cache = result_list.append

        def _line_cache_and_writer(**kwargs):
            """Write formatted line and cache arguments"""
            end = per_line % kwargs
            add_to_cache(end)
            write(start + end)
        return _line_cache_and_writer

    def _line_writer(**kwargs):
        """Write formatted line from arguments given by underlying opts"""
        write(start + per_line % kwargs)
    return _line_writer