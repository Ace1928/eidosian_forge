import numpy as np
from numpy import dot,  outer, random
from scipy import io, linalg, optimize
from scipy.sparse import eye as speye
import matplotlib.pyplot as plt
def Rp(v):
    """ Gradient """
    result = 2 * (A * v - R(v) * B * v) / dot(v.T, B * v)
    return result