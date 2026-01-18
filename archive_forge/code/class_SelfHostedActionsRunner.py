from __future__ import annotations
from typing import Any
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
class SelfHostedActionsRunner(NonCompletableGithubObject):
    """
    This class represents Self-hosted GitHub Actions Runners. The reference can be found at
    https://docs.github.com/en/free-pro-team@latest/rest/reference/actions#self-hosted-runners
    """

    def _initAttributes(self) -> None:
        self._id: Attribute[int] = NotSet
        self._name: Attribute[str] = NotSet
        self._os: Attribute[str] = NotSet
        self._status: Attribute[str] = NotSet
        self._busy: Attribute[bool] = NotSet
        self._labels: Attribute[list[dict[str, int | str]]] = NotSet

    def __repr__(self) -> str:
        return self.get__repr__({'name': self._name.value})

    @property
    def id(self) -> int:
        return self._id.value

    @property
    def name(self) -> str:
        return self._name.value

    @property
    def os(self) -> str:
        return self._os.value

    @property
    def status(self) -> str:
        return self._status.value

    @property
    def busy(self) -> bool:
        return self._busy.value

    def labels(self) -> list[dict[str, int | str]]:
        return self._labels.value

    def _useAttributes(self, attributes: dict[str, Any]) -> None:
        if 'id' in attributes:
            self._id = self._makeIntAttribute(attributes['id'])
        if 'name' in attributes:
            self._name = self._makeStringAttribute(attributes['name'])
        if 'os' in attributes:
            self._os = self._makeStringAttribute(attributes['os'])
        if 'status' in attributes:
            self._status = self._makeStringAttribute(attributes['status'])
        if 'busy' in attributes:
            self._busy = self._makeBoolAttribute(attributes['busy'])
        if 'labels' in attributes:
            self._labels = self._makeListOfDictsAttribute(attributes['labels'])