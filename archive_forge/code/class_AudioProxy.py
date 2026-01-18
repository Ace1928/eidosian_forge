from typing_extensions import override
from . import resources, _load_client
from ._utils import LazyProxy
class AudioProxy(LazyProxy[resources.Audio]):

    @override
    def __load__(self) -> resources.Audio:
        return _load_client().audio