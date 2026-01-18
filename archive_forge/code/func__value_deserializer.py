from __future__ import annotations
import hashlib
import json
import uuid
from functools import partial
from typing import Callable, List, Optional, Sequence, Union, cast
from langchain_core.embeddings import Embeddings
from langchain_core.stores import BaseStore, ByteStore
from langchain_core.utils.iter import batch_iterate
from langchain.storage.encoder_backed import EncoderBackedStore
def _value_deserializer(serialized_value: bytes) -> List[float]:
    """Deserialize a value."""
    return cast(List[float], json.loads(serialized_value.decode()))