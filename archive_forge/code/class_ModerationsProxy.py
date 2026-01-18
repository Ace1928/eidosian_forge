from typing_extensions import override
from . import resources, _load_client
from ._utils import LazyProxy
class ModerationsProxy(LazyProxy[resources.Moderations]):

    @override
    def __load__(self) -> resources.Moderations:
        return _load_client().moderations