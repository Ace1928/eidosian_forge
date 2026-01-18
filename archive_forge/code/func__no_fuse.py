from functools import partial
import numpy as np
from pennylane.math import abs as math_abs
from pennylane.math import sum as math_sum
from pennylane.math import allclose, arccos, arctan2, cos, get_interface, is_abstract, sin, stack
from pennylane.wires import Wires
from pennylane.ops.identity import GlobalPhase
def _no_fuse(angles_1, angles_2):
    """Special case: do not perform fusion when both Y angles are zero:
        Rot(a, 0, b) Rot(c, 0, d) = Rot(a + b + c + d, 0, 0)
    The quaternion math itself will fail in this case without a conditional.
    """
    return stack([angles_1[0] + angles_1[2] + angles_2[0] + angles_2[2], 0.0, 0.0])