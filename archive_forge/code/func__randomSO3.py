import numpy as np
import PIL.Image
import pythreejs
import scipy.interpolate
import ipyvolume as ipv
from ipyvolume.datasets import UrlCached
def _randomSO3():
    """Return random rotatation matrix, algo by James Arvo."""
    u1 = np.random.random()
    u2 = np.random.random()
    u3 = np.random.random()
    R = np.array([[np.cos(2 * np.pi * u1), np.sin(2 * np.pi * u1), 0], [-np.sin(2 * np.pi * u1), np.cos(2 * np.pi * u1), 0], [0, 0, 1]])
    v = np.array([np.cos(2 * np.pi * u2) * np.sqrt(u3), np.sin(2 * np.pi * u2) * np.sqrt(u3), np.sqrt(1 - u3)])
    H = np.identity(3) - 2 * v * np.transpose([v])
    return -np.dot(H, R)