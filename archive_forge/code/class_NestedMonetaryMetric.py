from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Tuple, Dict, Iterator, Any, Type, Set, Iterable, TYPE_CHECKING
from .debug import get_autologger
class NestedMonetaryMetric(BaseNestedMetric):
    """
    Nested Monetary Metric Container
    {
        'operation_a': {
            'total': 100.0,
            'average': 50.0,
            'median': 25.0,
            'count': 10,
        },
        'operation_b': {
            'total': 200.0,
            'average': 100.0,
            'median': 50.0,
            'count': 10,
        }
    }
    """
    name: Optional[str] = 'nested_monetary'
    data: Dict[str, MonetaryMetric] = Field(default_factory=dict, description='The nested monetary metric data')

    @property
    def metric_class(self) -> Type[MonetaryMetric]:
        """
        Returns the metric class
        """
        return MonetaryMetric

    @property
    def data_values(self) -> Dict[str, float]:
        """
        Returns the data values

        {
            'operation_a': {
                'total': 100.0,
                'average': 50.0,
                'median': 25.0,
                'count': 10,
            },
            'operation_b': {
                'total': 200.0,
                'average': 100.0,
                'median': 50.0,
                'count': 10,
            }
        }
        """
        return {k: v.data_values for k, v in self.data.items()}

    def items(self, **kwargs):
        """
        Returns the dict_items view of the data
        """
        return self.data_values.items()
    if TYPE_CHECKING:

        def __getitem__(self, key: str) -> MonetaryMetric:
            """
            Gets the value for the given key
            """
            ...

        def __getattr__(self, name: str) -> MonetaryMetric:
            """
            Gets the value for the given key
            """
            ...