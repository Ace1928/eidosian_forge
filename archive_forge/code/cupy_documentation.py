import numpy as np
from ..sharing import to_backend_cache_wrap
Convert constant arguments to cupy arrays, and perform any possible
    constant contractions.
    