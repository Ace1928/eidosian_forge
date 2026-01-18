from datetime import datetime
from typing import Any, Dict, Optional
import github.GithubObject
from github.GithubObject import Attribute, NotSet
class HookDeliveryResponse(github.GithubObject.NonCompletableGithubObject):
    """
    This class represents a HookDeliveryResponse
    """

    def __repr__(self) -> str:
        return self.get__repr__({'payload': self._payload.value})

    @property
    def headers(self) -> Optional[dict]:
        return self._response_headers.value

    @property
    def payload(self) -> Optional[str]:
        return self._payload.value

    def _initAttributes(self) -> None:
        self._response_headers: Attribute[Dict] = NotSet
        self._payload: Attribute[str] = NotSet

    def _useAttributes(self, attributes: Dict[str, Any]) -> None:
        if 'headers' in attributes:
            self._response_headers = self._makeDictAttribute(attributes['headers'])
        if 'payload' in attributes:
            self._payload = self._makeStringAttribute(attributes['payload'])