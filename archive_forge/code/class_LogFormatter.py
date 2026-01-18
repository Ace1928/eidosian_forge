import codecs
import itertools
import re
import sys
from io import BytesIO
from typing import Callable, Dict, List
from warnings import warn
from .lazy_import import lazy_import
from breezy import (
from breezy.i18n import gettext, ngettext
from . import errors, registry
from . import revision as _mod_revision
from . import revisionspec, trace
from . import transport as _mod_transport
from .osutils import (format_date,
from .tree import InterTree, find_previous_path
class LogFormatter:
    """Abstract class to display log messages.

    At a minimum, a derived class must implement the log_revision method.

    If the LogFormatter needs to be informed of the beginning or end of
    a log it should implement the begin_log and/or end_log hook methods.

    A LogFormatter should define the following supports_XXX flags
    to indicate which LogRevision attributes it supports:

    - supports_delta must be True if this log formatter supports delta.
      Otherwise the delta attribute may not be populated.  The 'delta_format'
      attribute describes whether the 'short_status' format (1) or the long
      one (2) should be used.

    - supports_merge_revisions must be True if this log formatter supports
      merge revisions.  If not, then only mainline revisions will be passed
      to the formatter.

    - preferred_levels is the number of levels this formatter defaults to.
      The default value is zero meaning display all levels.
      This value is only relevant if supports_merge_revisions is True.

    - supports_tags must be True if this log formatter supports tags.
      Otherwise the tags attribute may not be populated.

    - supports_diff must be True if this log formatter supports diffs.
      Otherwise the diff attribute may not be populated.

    - supports_signatures must be True if this log formatter supports GPG
      signatures.

    Plugins can register functions to show custom revision properties using
    the properties_handler_registry. The registered function
    must respect the following interface description::

        def my_show_properties(properties_dict):
            # code that returns a dict {'name':'value'} of the properties
            # to be shown
    """
    preferred_levels = 0

    def __init__(self, to_file, show_ids=False, show_timezone='original', delta_format=None, levels=None, show_advice=False, to_exact_file=None, author_list_handler=None):
        """Create a LogFormatter.

        :param to_file: the file to output to
        :param to_exact_file: if set, gives an output stream to which
             non-Unicode diffs are written.
        :param show_ids: if True, revision-ids are to be displayed
        :param show_timezone: the timezone to use
        :param delta_format: the level of delta information to display
          or None to leave it to the formatter to decide
        :param levels: the number of levels to display; None or -1 to
          let the log formatter decide.
        :param show_advice: whether to show advice at the end of the
          log or not
        :param author_list_handler: callable generating a list of
          authors to display for a given revision
        """
        self.to_file = to_file
        if to_exact_file is not None:
            self.to_exact_file = to_exact_file
        else:
            self.to_exact_file = getattr(to_file, 'stream', to_file)
        self.show_ids = show_ids
        self.show_timezone = show_timezone
        if delta_format is None:
            delta_format = 2
        self.delta_format = delta_format
        self.levels = levels
        self._show_advice = show_advice
        self._merge_count = 0
        self._author_list_handler = author_list_handler

    def get_levels(self):
        """Get the number of levels to display or 0 for all."""
        if getattr(self, 'supports_merge_revisions', False):
            if self.levels is None or self.levels == -1:
                self.levels = self.preferred_levels
        else:
            self.levels = 1
        return self.levels

    def log_revision(self, revision):
        """Log a revision.

        :param  revision:   The LogRevision to be logged.
        """
        raise NotImplementedError('not implemented in abstract base')

    def show_advice(self):
        """Output user advice, if any, when the log is completed."""
        if self._show_advice and self.levels == 1 and (self._merge_count > 0):
            advice_sep = self.get_advice_separator()
            if advice_sep:
                self.to_file.write(advice_sep)
            self.to_file.write('Use --include-merged or -n0 to see merged revisions.\n')

    def get_advice_separator(self):
        """Get the text separating the log from the closing advice."""
        return ''

    def short_committer(self, rev):
        name, address = config.parse_username(rev.committer)
        if name:
            return name
        return address

    def short_author(self, rev):
        return self.authors(rev, 'first', short=True, sep=', ')

    def authors(self, rev, who, short=False, sep=None):
        """Generate list of authors, taking --authors option into account.

        The caller has to specify the name of a author list handler,
        as provided by the author list registry, using the ``who``
        argument.  That name only sets a default, though: when the
        user selected a different author list generation using the
        ``--authors`` command line switch, as represented by the
        ``author_list_handler`` constructor argument, that value takes
        precedence.

        :param rev: The revision for which to generate the list of authors.
        :param who: Name of the default handler.
        :param short: Whether to shorten names to either name or address.
        :param sep: What separator to use for automatic concatenation.
        """
        if self._author_list_handler is not None:
            author_list_handler = self._author_list_handler
        else:
            author_list_handler = author_list_registry.get(who)
        names = author_list_handler(rev)
        if short:
            for i in range(len(names)):
                name, address = config.parse_username(names[i])
                if name:
                    names[i] = name
                else:
                    names[i] = address
        if sep is not None:
            names = sep.join(names)
        return names

    def merge_marker(self, revision):
        """Get the merge marker to include in the output or '' if none."""
        if len(revision.rev.parent_ids) > 1:
            self._merge_count += 1
            return ' [merge]'
        else:
            return ''

    def show_properties(self, revision, indent):
        """Displays the custom properties returned by each registered handler.

        If a registered handler raises an error it is propagated.
        """
        for line in self.custom_properties(revision):
            self.to_file.write('{}{}\n'.format(indent, line))

    def custom_properties(self, revision):
        """Format the custom properties returned by each registered handler.

        If a registered handler raises an error it is propagated.

        :return: a list of formatted lines (excluding trailing newlines)
        """
        lines = self._foreign_info_properties(revision)
        for key, handler in properties_handler_registry.iteritems():
            try:
                lines.extend(self._format_properties(handler(revision)))
            except Exception:
                trace.log_exception_quietly()
                trace.print_exception(sys.exc_info(), self.to_file)
        return lines

    def _foreign_info_properties(self, rev):
        """Custom log displayer for foreign revision identifiers.

        :param rev: Revision object.
        """
        if isinstance(rev, foreign.ForeignRevision):
            return self._format_properties(rev.mapping.vcs.show_foreign_revid(rev.foreign_revid))
        if b':' not in rev.revision_id:
            return []
        try:
            foreign_revid, mapping = foreign.foreign_vcs_registry.parse_revision_id(rev.revision_id)
        except errors.InvalidRevisionId:
            return []
        return self._format_properties(mapping.vcs.show_foreign_revid(foreign_revid))

    def _format_properties(self, properties):
        lines = []
        for key, value in properties.items():
            lines.append(key + ': ' + value)
        return lines

    def show_diff(self, to_file, diff, indent):
        encoding = get_terminal_encoding()
        for l in diff.rstrip().split(b'\n'):
            to_file.write(indent + l.decode(encoding, 'ignore') + '\n')