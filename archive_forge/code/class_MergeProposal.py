import re
from typing import Optional, Type
from . import errors, hooks, registry, urlutils
class MergeProposal:
    """A merge proposal.

    :ivar url: URL for the merge proposal
    """
    supports_auto_merge: bool

    def __init__(self, url=None):
        self.url = url

    def get_web_url(self):
        raise NotImplementedError(self.get_web_url)

    def get_description(self):
        """Get the description of the merge proposal."""
        raise NotImplementedError(self.get_description)

    def set_description(self, description):
        """Set the description of the merge proposal."""
        raise NotImplementedError(self.set_description)

    def get_title(self):
        """Get the title."""
        raise NotImplementedError(self.get_title)

    def set_title(self, title):
        """Get the title."""
        raise NotImplementedError(self.set_title)

    def get_commit_message(self):
        """Get the proposed commit message."""
        raise NotImplementedError(self.get_commit_message)

    def set_commit_message(self, commit_message):
        """Set the propose commit message."""
        raise NotImplementedError(self.set_commit_message)

    def get_source_branch_url(self, *, preferred_schemes=None):
        """Return the source branch."""
        raise NotImplementedError(self.get_source_branch_url)

    def get_source_revision(self):
        """Return the latest revision for the source branch."""
        raise NotImplementedError(self.get_source_revision)

    def get_target_branch_url(self, *, preferred_schemes=None):
        """Return the target branch."""
        raise NotImplementedError(self.get_target_branch_url)

    def set_target_branch_name(self):
        """Set the target branch name."""
        raise NotImplementedError(self.set_target_branch_name)

    def get_source_project(self):
        raise NotImplementedError(self.get_source_project)

    def get_target_project(self):
        raise NotImplementedError(self.get_target_project)

    def close(self):
        """Close the merge proposal (without merging it)."""
        raise NotImplementedError(self.close)

    def is_merged(self):
        """Check whether this merge proposal has been merged."""
        raise NotImplementedError(self.is_merged)

    def is_closed(self):
        """Check whether this merge proposal is closed

        This can either mean that it is merged or rejected.
        """
        raise NotImplementedError(self.is_closed)

    def merge(self, commit_message=None, auto=False):
        """Merge this merge proposal."""
        raise NotImplementedError(self.merge)

    def can_be_merged(self):
        """Can this merge proposal be merged?

        The answer to this can be no if e.g. it has conflics.
        """
        raise NotImplementedError(self.can_be_merged)

    def get_merged_by(self):
        """If this proposal was merged, who merged it.
        """
        raise NotImplementedError(self.get_merged_by)

    def get_merged_at(self):
        """If this proposal was merged, when it was merged.
        """
        raise NotImplementedError(self.get_merged_at)

    def post_comment(self, body):
        """Post a comment on the merge proposal.

        Args:
          body: Body of the comment
        """
        raise NotImplementedError(self.post_comment)

    def reopen(self):
        """Reopen this merge proposal."""
        raise NotImplementedError(self.reopen)