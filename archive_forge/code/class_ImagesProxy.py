from typing_extensions import override
from . import resources, _load_client
from ._utils import LazyProxy
class ImagesProxy(LazyProxy[resources.Images]):

    @override
    def __load__(self) -> resources.Images:
        return _load_client().images