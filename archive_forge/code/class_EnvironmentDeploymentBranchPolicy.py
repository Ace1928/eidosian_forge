from typing import Any, Dict
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
class EnvironmentDeploymentBranchPolicy(NonCompletableGithubObject):
    """
    This class represents a deployment branch policy for an environment. The reference can be found here https://docs.github.com/en/rest/reference/deployments#environments
    """

    def _initAttributes(self) -> None:
        self._protected_branches: Attribute[bool] = NotSet
        self._custom_branch_policies: Attribute[bool] = NotSet

    def __repr__(self) -> str:
        return self.get__repr__({})

    @property
    def protected_branches(self) -> bool:
        return self._protected_branches.value

    @property
    def custom_branch_policies(self) -> bool:
        return self._custom_branch_policies.value

    def _useAttributes(self, attributes: Dict[str, Any]) -> None:
        if 'protected_branches' in attributes:
            self._protected_branches = self._makeBoolAttribute(attributes['protected_branches'])
        if 'custom_branch_policies' in attributes:
            self._custom_branch_policies = self._makeBoolAttribute(attributes['custom_branch_policies'])