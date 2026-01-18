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
class _Outputter:
    """Precalculate formatting based on options given

    The idea here is to do this work only once per run, and finally return a
    function that will do the minimum amount possible for each match.
    """

    def __init__(self, opts, use_cache=False):
        self.outf = opts.outf
        if use_cache:
            self.cache = {}
        else:
            self.cache = None
        no_line = opts.files_with_matches or opts.files_without_match
        if opts.show_color:
            if no_line:
                self.get_writer = self._get_writer_plain
            elif opts.fixed_string:
                self._old = opts.pattern
                self._new = color_string(opts.pattern, FG.BOLD_RED)
                self.get_writer = self._get_writer_fixed_highlighted
            else:
                flags = opts.patternc.flags
                self._sub = re.compile(opts.pattern.join(('((?:', ')+)')), flags).sub
                self._highlight = color_string('\\1', FG.BOLD_RED)
                self.get_writer = self._get_writer_regexp_highlighted
            path_start = FG.MAGENTA
            path_end = FG.NONE
            sep = color_string(':', FG.BOLD_CYAN)
            rev_sep = color_string('~', FG.BOLD_YELLOW)
        else:
            self.get_writer = self._get_writer_plain
            path_start = path_end = ''
            sep = ':'
            rev_sep = '~'
        parts = [path_start, '%(path)s']
        if opts.print_revno:
            parts.extend([rev_sep, '%(revno)s'])
        self._format_initial = ''.join(parts)
        parts = []
        if no_line:
            if not opts.print_revno:
                parts.append(path_end)
        else:
            if opts.line_number:
                parts.extend([sep, '%(lineno)s'])
            parts.extend([sep, '%(line)s'])
        parts.append(opts.eol_marker)
        self._format_perline = ''.join(parts)

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

    def write_cached_lines(self, cache_id, revno):
        """Write cached results out again for new revision"""
        cached_path, cached_matches = self.cache[cache_id]
        start = self._format_initial % {'path': cached_path, 'revno': revno}
        write = self.outf.write
        for end in cached_matches:
            write(start + end)

    def _get_writer_regexp_highlighted(self, path, revno, cache_id):
        """Get function for writing output with regexp match highlighted"""
        _line_writer = self._get_writer_plain(path, revno, cache_id)
        sub, highlight = (self._sub, self._highlight)

        def _line_writer_regexp_highlighted(line, **kwargs):
            """Write formatted line with matched pattern highlighted"""
            return _line_writer(line=sub(highlight, line), **kwargs)
        return _line_writer_regexp_highlighted

    def _get_writer_fixed_highlighted(self, path, revno, cache_id):
        """Get function for writing output with search string highlighted"""
        _line_writer = self._get_writer_plain(path, revno, cache_id)
        old, new = (self._old, self._new)

        def _line_writer_fixed_highlighted(line, **kwargs):
            """Write formatted line with string searched for highlighted"""
            return _line_writer(line=line.replace(old, new), **kwargs)
        return _line_writer_fixed_highlighted