from typing import Any, Callable, Optional, Tuple
import warnings
import numpy as np
from scipy.stats import uniform, binom
def _get_ecdf_points(sample: np.ndarray, eval_points: np.ndarray, difference: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the coordinates for the ecdf points using compute_ecdf."""
    x = eval_points
    y = compute_ecdf(sample, eval_points)
    if not difference and y[0] > 0:
        x = np.insert(x, 0, x[0])
        y = np.insert(y, 0, 0)
    return (x, y)