import numpy as np
from scipy import stats
Example for estimating distribution parameters when some are fixed.

This uses currently a patched version of the distributions, two methods are
added to the continuous distributions. This has no side effects.
It also adds bounds to vonmises, which changes the behavior of it for some
methods.

