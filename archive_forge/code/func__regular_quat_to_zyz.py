from functools import partial
import numpy as np
from pennylane.math import abs as math_abs
from pennylane.math import sum as math_sum
from pennylane.math import allclose, arccos, arctan2, cos, get_interface, is_abstract, sin, stack
from pennylane.wires import Wires
from pennylane.ops.identity import GlobalPhase
def _regular_quat_to_zyz(qw, qx, qy, qz, y_arg):
    """Compute the ZYZ angles for the regular case (qx != 0 or qy != 0)"""
    z1_arg1 = 2 * (qy * qz - qw * qx)
    z1_arg2 = 2 * (qx * qz + qw * qy)
    z1 = arctan2(z1_arg1, z1_arg2)
    y = arccos(y_arg)
    z2_arg1 = 2 * (qy * qz + qw * qx)
    z2_arg2 = 2 * (qw * qy - qx * qz)
    z2 = arctan2(z2_arg1, z2_arg2)
    return stack([z1, y, z2])