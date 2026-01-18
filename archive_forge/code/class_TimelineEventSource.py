from __future__ import annotations
from typing import TYPE_CHECKING, Any
import github.Issue
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
class TimelineEventSource(NonCompletableGithubObject):
    """
    This class represents IssueTimelineEventSource. The reference can be found here https://docs.github.com/en/rest/reference/issues#timeline
    """

    def _initAttributes(self) -> None:
        self._type: Attribute[str] = NotSet
        self._issue: Attribute[Issue] = NotSet

    def __repr__(self) -> str:
        return self.get__repr__({'type': self._type.value})

    @property
    def type(self) -> str:
        return self._type.value

    @property
    def issue(self) -> Issue:
        return self._issue.value

    def _useAttributes(self, attributes: dict[str, Any]) -> None:
        if 'type' in attributes:
            self._type = self._makeStringAttribute(attributes['type'])
        if 'issue' in attributes:
            self._issue = self._makeClassAttribute(github.Issue.Issue, attributes['issue'])