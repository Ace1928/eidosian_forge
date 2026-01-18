from typing_extensions import override
from . import resources, _load_client
from ._utils import LazyProxy
class BatchesProxy(LazyProxy[resources.Batches]):

    @override
    def __load__(self) -> resources.Batches:
        return _load_client().batches