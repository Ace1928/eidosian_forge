import json
import numbers
from typing import Any
from pennylane.wires import Wires
Converts ``wires`` to a JSON list, with wire labels in
    order of their index.

    Returns:
        JSON list of wires

    Raises:
        UnserializableWireError: if any of the wires are not JSON-serializable.
    