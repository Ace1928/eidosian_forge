from typing import Any, Dict, Optional, Tuple
from wandb.data_types import Table
from wandb.errors import Error
class Visualize:

    def __init__(self, id: str, data: Table) -> None:
        self._id = id
        self._data = data

    def get_config_value(self, key: str) -> Dict[str, Any]:
        return {'id': self._id, 'historyFieldSettings': {'x-axis': '_step', 'key': key}}

    @staticmethod
    def get_config_key(key: str) -> Tuple[str, str, str]:
        return ('_wandb', 'viz', key)

    @property
    def value(self) -> Table:
        return self._data