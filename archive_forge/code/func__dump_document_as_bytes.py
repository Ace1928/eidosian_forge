from typing import Callable, Optional
from langchain_core.documents import Document
from langchain_core.load import Serializable, dumps, loads
from langchain_core.stores import BaseStore, ByteStore
from langchain.storage.encoder_backed import EncoderBackedStore
def _dump_document_as_bytes(obj: Document) -> bytes:
    """Return a bytes representation of a document."""
    if not isinstance(obj, Document):
        raise TypeError('Expected a Document instance')
    return dumps(obj).encode('utf-8')