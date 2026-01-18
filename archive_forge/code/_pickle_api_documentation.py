from typing import Optional
from . import cloudpickle
from .util import JSONInput, JSONOutput
Deserialize bytes with pickle.

    data (bytes): The data to deserialize.
    RETURNS: The deserialized Python object.
    