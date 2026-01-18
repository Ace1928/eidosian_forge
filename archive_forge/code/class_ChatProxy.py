from typing_extensions import override
from . import resources, _load_client
from ._utils import LazyProxy
class ChatProxy(LazyProxy[resources.Chat]):

    @override
    def __load__(self) -> resources.Chat:
        return _load_client().chat