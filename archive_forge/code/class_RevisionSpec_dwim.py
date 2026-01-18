from typing import List, Optional, Type
from breezy import revision, workingtree
from breezy.i18n import gettext
from . import errors, lazy_regex, registry
from . import revision as _mod_revision
from . import trace
class RevisionSpec_dwim(RevisionSpec):
    """Provides a DWIMish revision specifier lookup.

    Note that this does not go in the revspec_registry because by definition
    there is no prefix to identify it.  It's solely called from
    RevisionSpec.from_string() because the DWIMification happen when _match_on
    is called so the string describing the revision is kept here until needed.
    """
    help_txt: str
    _revno_regex = lazy_regex.lazy_compile('^(?:(\\d+(\\.\\d+)*)|-\\d+)(:.*)?$')
    _possible_revspecs: List[Type[registry._ObjectGetter]] = []

    def _try_spectype(self, rstype, branch):
        rs = rstype(self.spec, _internal=True)
        return rs.in_history(branch)

    def _match_on(self, branch, revs):
        """Run the lookup and see what we can get."""
        if self._revno_regex.match(self.spec) is not None:
            try:
                return self._try_spectype(RevisionSpec_revno, branch)
            except tuple(RevisionSpec_revno.dwim_catchable_exceptions):
                pass
        for objgetter in self._possible_revspecs:
            rs_class = objgetter.get_obj()
            try:
                return self._try_spectype(rs_class, branch)
            except tuple(rs_class.dwim_catchable_exceptions):
                pass
        raise InvalidRevisionSpec(self.spec, branch)

    @classmethod
    def append_possible_revspec(cls, revspec):
        """Append a possible DWIM revspec.

        :param revspec: Revision spec to try.
        """
        cls._possible_revspecs.append(registry._ObjectGetter(revspec))

    @classmethod
    def append_possible_lazy_revspec(cls, module_name, member_name):
        """Append a possible lazily loaded DWIM revspec.

        :param module_name: Name of the module with the revspec
        :param member_name: Name of the revspec within the module
        """
        cls._possible_revspecs.append(registry._LazyObjectGetter(module_name, member_name))