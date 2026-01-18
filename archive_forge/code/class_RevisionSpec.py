from typing import List, Optional, Type
from breezy import revision, workingtree
from breezy.i18n import gettext
from . import errors, lazy_regex, registry
from . import revision as _mod_revision
from . import trace
class RevisionSpec:
    """A parsed revision specification."""
    help_txt = 'A parsed revision specification.\n\n    A revision specification is a string, which may be unambiguous about\n    what it represents by giving a prefix like \'date:\' or \'revid:\' etc,\n    or it may have no prefix, in which case it\'s tried against several\n    specifier types in sequence to determine what the user meant.\n\n    Revision specs are an UI element, and they have been moved out\n    of the branch class to leave "back-end" classes unaware of such\n    details.  Code that gets a revno or rev_id from other code should\n    not be using revision specs - revnos and revision ids are the\n    accepted ways to refer to revisions internally.\n\n    (Equivalent to the old Branch method get_revision_info())\n    '
    prefix: Optional[str] = None
    dwim_catchable_exceptions: List[Type[Exception]] = [InvalidRevisionSpec]
    'Exceptions that RevisionSpec_dwim._match_on will catch.\n\n    If the revspec is part of ``dwim_revspecs``, it may be tried with an\n    invalid revspec and raises some exception. The exceptions mentioned here\n    will not be reported to the user but simply ignored without stopping the\n    dwim processing.\n    '

    @staticmethod
    def from_string(spec):
        """Parse a revision spec string into a RevisionSpec object.

        :param spec: A string specified by the user
        :return: A RevisionSpec object that understands how to parse the
            supplied notation.
        """
        if spec is None:
            return RevisionSpec(None, _internal=True)
        if not isinstance(spec, str):
            raise TypeError('revision spec needs to be text')
        match = revspec_registry.get_prefix(spec)
        if match is not None:
            spectype, specsuffix = match
            trace.mutter('Returning RevisionSpec %s for %s', spectype.__name__, spec)
            return spectype(spec, _internal=True)
        else:
            return RevisionSpec_dwim(spec, _internal=True)

    def __init__(self, spec, _internal=False):
        """Create a RevisionSpec referring to the Null revision.

        :param spec: The original spec supplied by the user
        :param _internal: Used to ensure that RevisionSpec is not being
            called directly. Only from RevisionSpec.from_string()
        """
        if not _internal:
            raise AssertionError('Creating a RevisionSpec directly is not supported. Use RevisionSpec.from_string() instead.')
        self.user_spec = spec
        if self.prefix and spec.startswith(self.prefix):
            spec = spec[len(self.prefix):]
        self.spec = spec

    def _match_on(self, branch, revs):
        trace.mutter('Returning RevisionSpec._match_on: None')
        return RevisionInfo(branch, None, None)

    def _match_on_and_check(self, branch, revs):
        info = self._match_on(branch, revs)
        if info:
            return info
        elif info == (None, None):
            return info
        elif self.prefix:
            raise InvalidRevisionSpec(self.user_spec, branch)
        else:
            raise InvalidRevisionSpec(self.spec, branch)

    def in_history(self, branch):
        return self._match_on_and_check(branch, revs=None)
    in_store = in_history
    in_branch = in_store

    def as_revision_id(self, context_branch):
        """Return just the revision_id for this revisions spec.

        Some revision specs require a context_branch to be able to determine
        their value. Not all specs will make use of it.
        """
        return self._as_revision_id(context_branch)

    def _as_revision_id(self, context_branch):
        """Implementation of as_revision_id()

        Classes should override this function to provide appropriate
        functionality. The default is to just call '.in_history().rev_id'
        """
        return self.in_history(context_branch).rev_id

    def as_tree(self, context_branch):
        """Return the tree object for this revisions spec.

        Some revision specs require a context_branch to be able to determine
        the revision id and access the repository. Not all specs will make
        use of it.
        """
        return self._as_tree(context_branch)

    def _as_tree(self, context_branch):
        """Implementation of as_tree().

        Classes should override this function to provide appropriate
        functionality. The default is to just call '.as_revision_id()'
        and get the revision tree from context_branch's repository.
        """
        revision_id = self.as_revision_id(context_branch)
        return context_branch.repository.revision_tree(revision_id)

    def __repr__(self):
        return '<{} {}>'.format(self.__class__.__name__, self.user_spec)

    def needs_branch(self):
        """Whether this revision spec needs a branch.

        Set this to False the branch argument of _match_on is not used.
        """
        return True

    def get_branch(self):
        """When the revision specifier contains a branch location, return it.

        Otherwise, return None.
        """
        return None