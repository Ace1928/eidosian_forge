from typing import Any, Dict
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
class Permissions(NonCompletableGithubObject):
    """
    This class represents Permissions
    """

    def _initAttributes(self) -> None:
        self._admin: Attribute[bool] = NotSet
        self._maintain: Attribute[bool] = NotSet
        self._pull: Attribute[bool] = NotSet
        self._push: Attribute[bool] = NotSet
        self._triage: Attribute[bool] = NotSet

    def __repr__(self) -> str:
        return self.get__repr__({'admin': self._admin.value, 'maintain': self._maintain.value, 'pull': self._pull.value, 'push': self._push.value, 'triage': self._triage.value})

    @property
    def admin(self) -> bool:
        return self._admin.value

    @property
    def maintain(self) -> bool:
        return self._maintain.value

    @property
    def pull(self) -> bool:
        return self._pull.value

    @property
    def push(self) -> bool:
        return self._push.value

    @property
    def triage(self) -> bool:
        return self._triage.value

    def _useAttributes(self, attributes: Dict[str, Any]) -> None:
        if 'admin' in attributes:
            self._admin = self._makeBoolAttribute(attributes['admin'])
        if 'maintain' in attributes:
            self._maintain = self._makeBoolAttribute(attributes['maintain'])
        if 'pull' in attributes:
            self._pull = self._makeBoolAttribute(attributes['pull'])
        if 'push' in attributes:
            self._push = self._makeBoolAttribute(attributes['push'])
        if 'triage' in attributes:
            self._triage = self._makeBoolAttribute(attributes['triage'])