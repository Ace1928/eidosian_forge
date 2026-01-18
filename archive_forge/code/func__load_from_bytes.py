from typing import Callable, Optional
from langchain_core.documents import Document
from langchain_core.load import Serializable, dumps, loads
from langchain_core.stores import BaseStore, ByteStore
from langchain.storage.encoder_backed import EncoderBackedStore
def _load_from_bytes(serialized: bytes) -> Serializable:
    """Return a document from a bytes representation."""
    return loads(serialized.decode('utf-8'))