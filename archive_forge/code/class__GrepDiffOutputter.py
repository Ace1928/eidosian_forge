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
class _GrepDiffOutputter:
    """Precalculate formatting based on options given for diff grep.
    """

    def __init__(self, opts):
        self.opts = opts
        self.outf = opts.outf
        if opts.show_color:
            if opts.fixed_string:
                self._old = opts.pattern
                self._new = color_string(opts.pattern, FG.BOLD_RED)
                self.get_writer = self._get_writer_fixed_highlighted
            else:
                flags = opts.patternc.flags
                self._sub = re.compile(opts.pattern.join(('((?:', ')+)')), flags).sub
                self._highlight = color_string('\\1', FG.BOLD_RED)
                self.get_writer = self._get_writer_regexp_highlighted
        else:
            self.get_writer = self._get_writer_plain

    def get_file_header_writer(self):
        """Get function for writing file headers"""
        write = self.outf.write
        eol_marker = self.opts.eol_marker

        def _line_writer(line):
            write(line + eol_marker)

        def _line_writer_color(line):
            write(FG.BOLD_MAGENTA + line + FG.NONE + eol_marker)
        if self.opts.show_color:
            return _line_writer_color
        else:
            return _line_writer
        return _line_writer

    def get_revision_header_writer(self):
        """Get function for writing revno lines"""
        write = self.outf.write
        eol_marker = self.opts.eol_marker

        def _line_writer(line):
            write(line + eol_marker)

        def _line_writer_color(line):
            write(FG.BOLD_BLUE + line + FG.NONE + eol_marker)
        if self.opts.show_color:
            return _line_writer_color
        else:
            return _line_writer
        return _line_writer

    def _get_writer_plain(self):
        """Get function for writing uncoloured output"""
        write = self.outf.write
        eol_marker = self.opts.eol_marker

        def _line_writer(line):
            write(line + eol_marker)
        return _line_writer

    def _get_writer_regexp_highlighted(self):
        """Get function for writing output with regexp match highlighted"""
        _line_writer = self._get_writer_plain()
        sub, highlight = (self._sub, self._highlight)

        def _line_writer_regexp_highlighted(line):
            """Write formatted line with matched pattern highlighted"""
            return _line_writer(line=sub(highlight, line))
        return _line_writer_regexp_highlighted

    def _get_writer_fixed_highlighted(self):
        """Get function for writing output with search string highlighted"""
        _line_writer = self._get_writer_plain()
        old, new = (self._old, self._new)

        def _line_writer_fixed_highlighted(line):
            """Write formatted line with string searched for highlighted"""
            return _line_writer(line=line.replace(old, new))
        return _line_writer_fixed_highlighted