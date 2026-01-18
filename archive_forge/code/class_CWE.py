from typing import Any, Dict
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
class CWE(CompletableGithubObject):
    """
    This class represents a CWE.
    The reference can be found here https://docs.github.com/en/rest/security-advisories/repository-advisories
    """

    def _initAttributes(self) -> None:
        self._cwe_id: Attribute[str] = NotSet
        self._name: Attribute[str] = NotSet

    @property
    def cwe_id(self) -> str:
        return self._cwe_id.value

    @property
    def name(self) -> str:
        return self._name.value

    def _useAttributes(self, attributes: Dict[str, Any]) -> None:
        if 'cwe_id' in attributes:
            self._cwe_id = self._makeStringAttribute(attributes['cwe_id'])
        if 'name' in attributes:
            self._name = self._makeStringAttribute(attributes['name'])