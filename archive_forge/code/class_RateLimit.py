from __future__ import annotations
from typing import TYPE_CHECKING, Any
import github.Rate
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
class RateLimit(NonCompletableGithubObject):
    """
    This class represents RateLimits. The reference can be found here https://docs.github.com/en/rest/reference/rate-limit
    """

    def _initAttributes(self) -> None:
        self._core: Attribute[Rate] = NotSet
        self._search: Attribute[Rate] = NotSet
        self._graphql: Attribute[Rate] = NotSet

    def __repr__(self) -> str:
        return self.get__repr__({'core': self._core.value})

    @property
    def core(self) -> Rate:
        """
        Rate limit for the non-search-related API

        :type: class:`github.Rate.Rate`
        """
        return self._core.value

    @property
    def search(self) -> Rate:
        """
        Rate limit for the Search API.

        :type: class:`github.Rate.Rate`
        """
        return self._search.value

    @property
    def graphql(self) -> Rate:
        """
        (Experimental) Rate limit for GraphQL API, use with caution.

        :type: class:`github.Rate.Rate`
        """
        return self._graphql.value

    def _useAttributes(self, attributes: dict[str, Any]) -> None:
        if 'core' in attributes:
            self._core = self._makeClassAttribute(github.Rate.Rate, attributes['core'])
        if 'search' in attributes:
            self._search = self._makeClassAttribute(github.Rate.Rate, attributes['search'])
        if 'graphql' in attributes:
            self._graphql = self._makeClassAttribute(github.Rate.Rate, attributes['graphql'])