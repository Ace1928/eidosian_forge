from typing import Union
from .aggregation import Asc, Desc, Reducer, SortDirection
class random_sample(Reducer):
    """
    Returns a random sample of items from the dataset, from the given property
    """
    NAME = 'RANDOM_SAMPLE'

    def __init__(self, field: str, size: int) -> None:
        """
        ### Parameter

        **field**: Field to sample from
        **size**: Return this many items (can be less)
        """
        args = [field, str(size)]
        super().__init__(*args)
        self._field = field