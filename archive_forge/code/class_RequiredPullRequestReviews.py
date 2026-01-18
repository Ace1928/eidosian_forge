from __future__ import annotations
from typing import TYPE_CHECKING, Any
import github.NamedUser
import github.Team
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
class RequiredPullRequestReviews(CompletableGithubObject):
    """
    This class represents Required Pull Request Reviews. The reference can be found here https://docs.github.com/en/rest/reference/repos#get-pull-request-review-protection
    """

    def _initAttributes(self) -> None:
        self._dismiss_stale_reviews: Attribute[bool] = NotSet
        self._require_code_owner_reviews: Attribute[bool] = NotSet
        self._required_approving_review_count: Attribute[int] = NotSet
        self._users: Attribute[list[NamedUser]] = NotSet
        self._teams: Attribute[list[Team]] = NotSet
        self._require_last_push_approval: Attribute[bool] = NotSet

    def __repr__(self) -> str:
        return self.get__repr__({'url': self._url.value, 'dismiss_stale_reviews': self._dismiss_stale_reviews.value, 'require_code_owner_reviews': self._require_code_owner_reviews.value, 'require_last_push_approval': self._require_last_push_approval.value})

    @property
    def dismiss_stale_reviews(self) -> bool:
        self._completeIfNotSet(self._dismiss_stale_reviews)
        return self._dismiss_stale_reviews.value

    @property
    def require_code_owner_reviews(self) -> bool:
        self._completeIfNotSet(self._require_code_owner_reviews)
        return self._require_code_owner_reviews.value

    @property
    def required_approving_review_count(self) -> int:
        self._completeIfNotSet(self._required_approving_review_count)
        return self._required_approving_review_count.value

    @property
    def require_last_push_approval(self) -> bool:
        self._completeIfNotSet(self._require_last_push_approval)
        return self._require_last_push_approval.value

    @property
    def url(self) -> str:
        self._completeIfNotSet(self._url)
        return self._url.value

    @property
    def dismissal_users(self) -> list[NamedUser]:
        self._completeIfNotSet(self._users)
        return self._users.value

    @property
    def dismissal_teams(self) -> list[Team]:
        self._completeIfNotSet(self._teams)
        return self._teams.value

    def _useAttributes(self, attributes: dict[str, Any]) -> None:
        if 'dismissal_restrictions' in attributes:
            if 'users' in attributes['dismissal_restrictions']:
                self._users = self._makeListOfClassesAttribute(github.NamedUser.NamedUser, attributes['dismissal_restrictions']['users'])
            if 'teams' in attributes['dismissal_restrictions']:
                self._teams = self._makeListOfClassesAttribute(github.Team.Team, attributes['dismissal_restrictions']['teams'])
        if 'dismiss_stale_reviews' in attributes:
            self._dismiss_stale_reviews = self._makeBoolAttribute(attributes['dismiss_stale_reviews'])
        if 'require_code_owner_reviews' in attributes:
            self._require_code_owner_reviews = self._makeBoolAttribute(attributes['require_code_owner_reviews'])
        if 'required_approving_review_count' in attributes:
            self._required_approving_review_count = self._makeIntAttribute(attributes['required_approving_review_count'])
        if 'require_last_push_approval' in attributes:
            self._require_last_push_approval = self._makeBoolAttribute(attributes['require_last_push_approval'])
        if 'url' in attributes:
            self._url = self._makeStringAttribute(attributes['url'])