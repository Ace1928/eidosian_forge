from typing_extensions import override
from . import resources, _load_client
from ._utils import LazyProxy
class CompletionsProxy(LazyProxy[resources.Completions]):

    @override
    def __load__(self) -> resources.Completions:
        return _load_client().completions