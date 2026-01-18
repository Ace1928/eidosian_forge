import json
import numbers
from typing import Any
from pennylane.wires import Wires
class UnserializableWireError(TypeError):
    """Raised if a wire label is not JSON-serializable."""

    def __init__(self, wire: Any) -> None:
        super().__init__(f"Cannot serialize wire label '{wire}': Type '{type(wire)}' is not json-serializable.")