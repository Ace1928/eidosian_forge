import numpy as np
from numpy import dot,  outer, random
from scipy import io, linalg, optimize
from scipy.sparse import eye as speye
import matplotlib.pyplot as plt
def Rpp(v):
    """ Hessian """
    result = 2 * (A - R(v) * B - outer(B * v, Rp(v)) - outer(Rp(v), B * v)) / dot(v.T, B * v)
    return result