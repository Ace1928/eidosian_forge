from typing import Any, Callable, Dict, Generic, Iterator, List, Optional, Type, TypeVar, Union
from urllib.parse import parse_qs
from github.GithubObject import GithubObject
from github.Requester import Requester

    This class abstracts the `pagination of the API <https://docs.github.com/en/rest/guides/traversing-with-pagination>`_.

    You can simply enumerate through instances of this class::

        for repo in user.get_repos():
            print(repo.name)

    If you want to know the total number of items in the list::

        print(user.get_repos().totalCount)

    You can also index them or take slices::

        second_repo = user.get_repos()[1]
        first_repos = user.get_repos()[:10]

    If you want to iterate in reversed order, just do::

        for repo in user.get_repos().reversed:
            print(repo.name)

    And if you really need it, you can explicitly access a specific page::

        some_repos = user.get_repos().get_page(0)
        some_other_repos = user.get_repos().get_page(3)
    