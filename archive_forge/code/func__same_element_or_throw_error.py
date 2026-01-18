from typing import (
import numpy as np
from cirq import value
from cirq.circuits._block_diagram_drawer import BlockDiagramDrawer
from cirq.circuits._box_drawing_character_data import (
def _same_element_or_throw_error(elements: Sequence[Any]):
    """Extract an element or throw an error.

    Args:
        elements: A sequence of something.

    copies of it. Returns None on an empty sequence.

    Raises:
        ValueError: The sequence contains more than one unique element.

    Returns:
        The element when given a sequence containing only multiple copies of a
        single element. None if elements is empty.
    """
    unique_elements = set(elements)
    if len(unique_elements) > 1:
        raise ValueError(f'len(set({elements})) > 1')
    return unique_elements.pop() if elements else None