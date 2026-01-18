import time
from typing import Any, Callable, Iterable, List, Tuple, Union
import ray
from ray import ObjectRef
from ray.cluster_utils import Cluster
class ObjectStoreWriterNonStreaming(ObjectStoreWriter):

    def __init__(self):
        self.results = []

    def add(self, item: InType) -> None:
        self.results.append(item)

    def finish(self) -> List[Any]:
        return self.results