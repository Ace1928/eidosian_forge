import functools
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Callable, Iterator, Optional, Set, Tuple
from pip._vendor.packaging.version import _BaseVersion
from .base import Candidate
def _iter_built_with_prepended(installed: Candidate, infos: Iterator[IndexCandidateInfo]) -> Iterator[Candidate]:
    """Iterator for ``FoundCandidates``.

    This iterator is used when the resolver prefers the already-installed
    candidate and NOT to upgrade. The installed candidate is therefore
    always yielded first, and candidates from index come later in their
    normal ordering, except skipped when the version is already installed.
    """
    yield installed
    versions_found: Set[_BaseVersion] = {installed.version}
    for version, func in infos:
        if version in versions_found:
            continue
        candidate = func()
        if candidate is None:
            continue
        yield candidate
        versions_found.add(version)