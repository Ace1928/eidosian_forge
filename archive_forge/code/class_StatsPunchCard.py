from typing import Any, Dict, Tuple
import github.GithubObject
import github.NamedUser  # TODO remove unused
class StatsPunchCard(github.GithubObject.NonCompletableGithubObject):
    """
    This class represents StatsPunchCards. The reference can be found here https://docs.github.com/en/rest/reference/repos#get-the-hourly-commit-count-for-each-day
    """
    _dict: Dict[Tuple[int, int], int]

    def get(self, day: int, hour: int) -> int:
        """Get a specific element"""
        return self._dict[day, hour]

    def _initAttributes(self) -> None:
        self._dict = {}

    def _useAttributes(self, attributes: Any) -> None:
        for day, hour, commits in attributes:
            self._dict[day, hour] = commits