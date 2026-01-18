from fnmatch import fnmatch
from pathlib import Path
from typing import Callable, Generator, Iterable, List, Optional, TypeVar, Union
Filter repo objects based on an allowlist and a denylist.

    Input must be a list of paths (`str` or `Path`) or a list of arbitrary objects.
    In the later case, `key` must be provided and specifies a function of one argument
    that is used to extract a path from each element in iterable.

    Patterns are Unix shell-style wildcards which are NOT regular expressions. See
    https://docs.python.org/3/library/fnmatch.html for more details.

    Args:
        items (`Iterable`):
            List of items to filter.
        allow_patterns (`str` or `List[str]`, *optional*):
            Patterns constituting the allowlist. If provided, item paths must match at
            least one pattern from the allowlist.
        ignore_patterns (`str` or `List[str]`, *optional*):
            Patterns constituting the denylist. If provided, item paths must not match
            any patterns from the denylist.
        key (`Callable[[T], str]`, *optional*):
            Single-argument function to extract a path from each item. If not provided,
            the `items` must already be `str` or `Path`.

    Returns:
        Filtered list of objects, as a generator.

    Raises:
        :class:`ValueError`:
            If `key` is not provided and items are not `str` or `Path`.

    Example usage with paths:
    ```python
    >>> # Filter only PDFs that are not hidden.
    >>> list(filter_repo_objects(
    ...     ["aaa.PDF", "bbb.jpg", ".ccc.pdf", ".ddd.png"],
    ...     allow_patterns=["*.pdf"],
    ...     ignore_patterns=[".*"],
    ... ))
    ["aaa.pdf"]
    ```

    Example usage with objects:
    ```python
    >>> list(filter_repo_objects(
    ... [
    ...     CommitOperationAdd(path_or_fileobj="/tmp/aaa.pdf", path_in_repo="aaa.pdf")
    ...     CommitOperationAdd(path_or_fileobj="/tmp/bbb.jpg", path_in_repo="bbb.jpg")
    ...     CommitOperationAdd(path_or_fileobj="/tmp/.ccc.pdf", path_in_repo=".ccc.pdf")
    ...     CommitOperationAdd(path_or_fileobj="/tmp/.ddd.png", path_in_repo=".ddd.png")
    ... ],
    ... allow_patterns=["*.pdf"],
    ... ignore_patterns=[".*"],
    ... key=lambda x: x.repo_in_path
    ... ))
    [CommitOperationAdd(path_or_fileobj="/tmp/aaa.pdf", path_in_repo="aaa.pdf")]
    ```
    