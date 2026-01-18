from typing_extensions import override
from . import resources, _load_client
from ._utils import LazyProxy
class ModelsProxy(LazyProxy[resources.Models]):

    @override
    def __load__(self) -> resources.Models:
        return _load_client().models