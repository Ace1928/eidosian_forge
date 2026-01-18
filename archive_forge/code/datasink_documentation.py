from typing import Any, Iterable, List
from ray.data._internal.execution.interfaces import TaskContext
from ray.data.block import Block
from ray.util.annotations import DeveloperAPI
If ``False``, only launch write tasks on the driver's node.