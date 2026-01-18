import collections
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Union
import ray
from ray.data._internal.compute import ActorPoolStrategy
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.execution.interfaces import (
from ray.data._internal.execution.operators.map_operator import MapOperator, _map_task
from ray.data._internal.execution.operators.map_transformer import MapTransformer
from ray.data._internal.execution.util import locality_string
from ray.data.block import Block, BlockMetadata
from ray.data.context import DataContext
from ray.types import ObjectRef
def _get_location(self, bundle: RefBundle) -> Optional[NodeIdStr]:
    """Ask Ray for the node id of the given bundle.

        This method may be overriden for testing.

        Returns:
            A node id associated with the bundle, or None if unknown.
        """
    return bundle.get_cached_location()