from typing_extensions import override
from . import resources, _load_client
from ._utils import LazyProxy
class EmbeddingsProxy(LazyProxy[resources.Embeddings]):

    @override
    def __load__(self) -> resources.Embeddings:
        return _load_client().embeddings