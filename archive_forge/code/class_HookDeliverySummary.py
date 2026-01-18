from datetime import datetime
from typing import Any, Dict, Optional
import github.GithubObject
from github.GithubObject import Attribute, NotSet
class HookDeliverySummary(github.GithubObject.NonCompletableGithubObject):
    """
    This class represents a Summary of HookDeliveries
    """

    def _initAttributes(self) -> None:
        self._id: Attribute[int] = NotSet
        self._guid: Attribute[str] = NotSet
        self._delivered_at: Attribute[datetime] = NotSet
        self._redelivery: Attribute[bool] = NotSet
        self._duration: Attribute[float] = NotSet
        self._status: Attribute[str] = NotSet
        self._status_code: Attribute[int] = NotSet
        self._event: Attribute[str] = NotSet
        self._action: Attribute[str] = NotSet
        self._installation_id: Attribute[int] = NotSet
        self._repository_id: Attribute[int] = NotSet
        self._url: Attribute[str] = NotSet

    def __repr__(self) -> str:
        return self.get__repr__({'id': self._id.value})

    @property
    def id(self) -> Optional[int]:
        return self._id.value

    @property
    def guid(self) -> Optional[str]:
        return self._guid.value

    @property
    def delivered_at(self) -> Optional[datetime]:
        return self._delivered_at.value

    @property
    def redelivery(self) -> Optional[bool]:
        return self._redelivery.value

    @property
    def duration(self) -> Optional[float]:
        return self._duration.value

    @property
    def status(self) -> Optional[str]:
        return self._status.value

    @property
    def status_code(self) -> Optional[int]:
        return self._status_code.value

    @property
    def event(self) -> Optional[str]:
        return self._event.value

    @property
    def action(self) -> Optional[str]:
        return self._action.value

    @property
    def installation_id(self) -> Optional[int]:
        return self._installation_id.value

    @property
    def repository_id(self) -> Optional[int]:
        return self._repository_id.value

    @property
    def url(self) -> Optional[str]:
        return self._url.value

    def _useAttributes(self, attributes: Dict[str, Any]) -> None:
        if 'id' in attributes:
            self._id = self._makeIntAttribute(attributes['id'])
        if 'guid' in attributes:
            self._guid = self._makeStringAttribute(attributes['guid'])
        if 'delivered_at' in attributes:
            self._delivered_at = self._makeDatetimeAttribute(attributes['delivered_at'])
        if 'redelivery' in attributes:
            self._redelivery = self._makeBoolAttribute(attributes['redelivery'])
        if 'duration' in attributes:
            self._duration = self._makeFloatAttribute(attributes['duration'])
        if 'status' in attributes:
            self._status = self._makeStringAttribute(attributes['status'])
        if 'status_code' in attributes:
            self._status_code = self._makeIntAttribute(attributes['status_code'])
        if 'event' in attributes:
            self._event = self._makeStringAttribute(attributes['event'])
        if 'action' in attributes:
            self._action = self._makeStringAttribute(attributes['action'])
        if 'installation_id' in attributes:
            self._installation_id = self._makeIntAttribute(attributes['installation_id'])
        if 'repository_id' in attributes:
            self._repository_id = self._makeIntAttribute(attributes['repository_id'])
        if 'url' in attributes:
            self._url = self._makeStringAttribute(attributes['url'])