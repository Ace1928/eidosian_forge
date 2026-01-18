from typing import Dict, Iterable, List, Tuple
from torch import nn
from .namespace import Namespace
class SkipLayout:
    """Represents a skip connection layout across partitions."""
    by_ns_name: Dict[Tuple[Namespace, str], Tuple[int, int]]
    by_partition: List[List[Tuple[int, Namespace, str]]]

    def __init__(self, num_partitions: int, skip_routes: Dict[Tuple[Namespace, str], Tuple[int, int]]) -> None:
        self.by_ns_name = skip_routes
        self.by_partition = [[] for _ in range(num_partitions)]
        for (ns, name), (prev_j, next_j) in skip_routes.items():
            self.by_partition[next_j].append((prev_j, ns, name))
        for p in self.by_partition:
            p.sort()

    def copy_policy(self, next_j: int) -> Iterable[Tuple[int, Namespace, str]]:
        """Generates skip routes for the given destination partition number.
        The skip routes are sorted by source partition number in ascending
        order.

        Yields:
            Each tuple of (source partition number, namespace, name).

        """
        for prev_j, ns, name in self.by_partition[next_j]:
            if prev_j == next_j:
                continue
            yield (prev_j, ns, name)

    def requires_copy(self, ns: Namespace, name: str) -> bool:
        """Whether the given namespace and name requires partition-to-partition
        copy or not.
        """
        prev_j, next_j = self.by_ns_name.get((ns, name), (-1, -1))
        return prev_j != next_j