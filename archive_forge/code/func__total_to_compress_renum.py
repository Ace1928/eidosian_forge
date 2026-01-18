import numpy as np
from matplotlib import _api
from matplotlib.tri import Triangulation
@staticmethod
def _total_to_compress_renum(valid):
    """
        Parameters
        ----------
        valid : 1D bool array
            Validity mask.

        Returns
        -------
        int array
            Array so that (`valid_array` being a compressed array
            based on a `masked_array` with mask ~*valid*):

            - For all i with valid[i] = True:
              valid_array[renum[i]] = masked_array[i]
            - For all i with valid[i] = False:
              renum[i] = -1 (invalid value)
        """
    renum = np.full(np.size(valid), -1, dtype=np.int32)
    n_valid = np.sum(valid)
    renum[valid] = np.arange(n_valid, dtype=np.int32)
    return renum