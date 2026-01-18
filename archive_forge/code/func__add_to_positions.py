from functools import lru_cache
from typing import Sequence, Dict, Union, Tuple, List, Optional
import numpy as np
import quimb
import quimb.tensor as qtn
import cirq
def _add_to_positions(positions: Dict[Tuple[str, str], Tuple[float, float]], mi: int, qubits: Union[cirq.Qid, Tuple[cirq.Qid]], *, all_qubits: Tuple[cirq.Qid, ...], x_scale, y_scale, x_nudge, yb_offset):
    """Helper function to update the `positions` dictionary.

    Args:
        positions: The dictionary to update. Quimb will consume this for drawing
        mi: Moment index (used for x-positioning)
        qubits: The qubits (used for y-positioning)
        all_qubits: All qubits in the circuit, allowing us
            to position the zero'th qubit at the top.
        x_scale: Stretch coordinates in the x direction
        y_scale: Stretch coordinates in the y direction
        x_nudge: Kraus operators will have vertical lines connecting the
            "forward" and "backward" circuits, so the x position of each
            tensor is nudged (according to its y position) to help see all
            the lines.
        yb_offset: Offset the "backwards" circuit by this much.
    """
    qy = _qpos_y(qubits, all_qubits)
    positions[f'i{mi}f', _qpos_tag(qubits)] = (mi * x_scale + qy * x_nudge, y_scale * qy)
    positions[f'i{mi}b', _qpos_tag(qubits)] = (mi * x_scale, y_scale * qy + yb_offset)