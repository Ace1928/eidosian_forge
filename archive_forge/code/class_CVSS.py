from decimal import Decimal
from typing import Any, Dict
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
class CVSS(NonCompletableGithubObject):
    """
    This class represents a CVSS.
    The reference can be found here <https://docs.github.com/en/rest/security-advisories/global-advisories>
    """

    def _initAttributes(self) -> None:
        self._vector_string: Attribute[str] = NotSet
        self._score: Attribute[Decimal] = NotSet
        self._version: Attribute[Decimal] = NotSet

    @property
    def score(self) -> Decimal:
        return self._score.value

    @property
    def version(self) -> Decimal:
        return self._version.value

    @property
    def vector_string(self) -> str:
        return self._vector_string.value

    def _useAttributes(self, attributes: Dict[str, Any]) -> None:
        if 'score' in attributes and attributes['score'] is not None:
            self._score = self._makeDecimalAttribute(Decimal(str(attributes['score'])))
        if 'vector_string' in attributes and attributes['vector_string'] is not None:
            self._vector_string = self._makeStringAttribute(attributes['vector_string'])
            self._version = self._makeDecimalAttribute(Decimal(self.vector_string.split(':')[1].split('/')[0]))