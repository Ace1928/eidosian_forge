from datetime import datetime
from typing import Any, Dict
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
class Rate(NonCompletableGithubObject):
    """
    This class represents Rates. The reference can be found here https://docs.github.com/en/rest/reference/rate-limit
    """

    def _initAttributes(self) -> None:
        self._limit: Attribute[int] = NotSet
        self._remaining: Attribute[int] = NotSet
        self._reset: Attribute[datetime] = NotSet
        self._used: Attribute[int] = NotSet

    def __repr__(self) -> str:
        return self.get__repr__({'limit': self._limit.value, 'remaining': self._remaining.value, 'reset': self._reset.value})

    @property
    def limit(self) -> int:
        return self._limit.value

    @property
    def remaining(self) -> int:
        return self._remaining.value

    @property
    def reset(self) -> datetime:
        return self._reset.value

    @property
    def used(self) -> int:
        return self._used.value

    def _useAttributes(self, attributes: Dict[str, Any]) -> None:
        if 'limit' in attributes:
            self._limit = self._makeIntAttribute(attributes['limit'])
        if 'remaining' in attributes:
            self._remaining = self._makeIntAttribute(attributes['remaining'])
        if 'reset' in attributes:
            self._reset = self._makeTimestampAttribute(attributes['reset'])
        if 'used' in attributes:
            self._used = self._makeIntAttribute(attributes['used'])