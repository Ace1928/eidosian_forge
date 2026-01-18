from functools import partial
import numpy as np
def _prism_red(x):
    return 0.75 * np.sin((x * 20.9 + 0.25) * np.pi) + 0.67