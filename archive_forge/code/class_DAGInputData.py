from typing import Any, Dict, List, Union, Optional
from ray.dag import DAGNode
from ray.dag.format_utils import get_dag_node_str
from ray.experimental.gradio_utils import type_to_string
from ray.util.annotations import DeveloperAPI
@DeveloperAPI
class DAGInputData:
    """If user passed multiple args and kwargs directly to dag.execute(), we
    generate this wrapper for all user inputs as one object, accessible via
    list index or object attribute key.
    """

    def __init__(self, *args, **kwargs):
        self._args = list(args)
        self._kwargs = kwargs

    def __getitem__(self, key: Union[int, str]) -> Any:
        if isinstance(key, int):
            return self._args[key]
        elif isinstance(key, str):
            return self._kwargs[key]
        else:
            raise ValueError('Please only use int index or str as first-level key to access fields of dag input.')