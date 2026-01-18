import dataclasses
import itertools
from typing import Any, cast, Iterator, List, Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from cirq import circuits, ops, protocols
@dataclasses.dataclass
class Cliffords:
    """The single-qubit Clifford group, decomposed into elementary gates.

    The decomposition of the Cliffords follows those described in
    Barends et al., Nature 508, 500 (https://arxiv.org/abs/1402.4848).

    Decompositions of the Clifford group:
        c1_in_xy: decomposed into XPowGate and YPowGate.
        c1_in_xz: decomposed into XPowGate and ZPowGate, with at most one
            XPowGate (one microwave gate) per Clifford.

    Subsets used to generate the 2-qubit Clifford group (see paper table S7):
        s1
        s1_x
        s1_y
    """
    c1_in_xy: List[List[ops.Gate]]
    c1_in_xz: List[List[ops.Gate]]
    s1: List[List[ops.Gate]]
    s1_x: List[List[ops.Gate]]
    s1_y: List[List[ops.Gate]]