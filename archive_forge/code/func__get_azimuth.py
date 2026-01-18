import numpy as np  # type: ignore
from typing import Tuple, Optional
def _get_azimuth(x: float, y: float) -> float:
    sign_y = -1.0 if y < 0.0 else 1.0
    sign_x = -1.0 if x < 0.0 else 1.0
    return np.arctan2(y, x) if 0 != x and 0 != y else np.pi / 2.0 * sign_y if 0 != y else np.pi if sign_x < 0.0 else 0.0