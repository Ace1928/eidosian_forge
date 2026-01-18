from collections.abc import Iterable
import operator
import warnings
import numpy
If input is a scalar, create a sequence of length equal to the
    rank by duplicating the input. If input is a sequence,
    check if its length is equal to the length of array.
    