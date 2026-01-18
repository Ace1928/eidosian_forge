import numpy as np
import math
from Algorithm import Algorithm
from Constants import USER_SEED
def ReLU(x):
    return x * (x > 0)