from *Random walks for image segmentation*, Leo Grady, IEEE Trans
import numpy as np
from scipy import sparse, ndimage as ndi
from .._shared import utils
from .._shared.utils import warn
from .._shared.compat import SCIPY_CG_TOL_PARAM_NAME
from ..util import img_as_float
from scipy.sparse.linalg import cg, spsolve
def _solve_linear_system(lap_sparse, B, tol, mode):
    if mode is None:
        mode = 'cg_j'
    if mode == 'cg_mg' and (not amg_loaded):
        warn('"cg_mg" not available, it requires pyamg to be installed. The "cg_j" mode will be used instead.', stacklevel=2)
        mode = 'cg_j'
    if mode == 'bf':
        X = spsolve(lap_sparse, B.toarray()).T
    else:
        maxiter = None
        if mode == 'cg':
            if UmfpackContext is None:
                warn('"cg" mode may be slow because UMFPACK is not available. Consider building Scipy with UMFPACK or use a preconditioned version of CG ("cg_j" or "cg_mg" modes).', stacklevel=2)
            M = None
        elif mode == 'cg_j':
            M = sparse.diags(1.0 / lap_sparse.diagonal())
        else:
            lap_sparse = lap_sparse.tocsr()
            ml = ruge_stuben_solver(lap_sparse, coarse_solver='pinv')
            M = ml.aspreconditioner(cycle='V')
            maxiter = 30
        rtol = {SCIPY_CG_TOL_PARAM_NAME: tol}
        cg_out = [cg(lap_sparse, B[:, i].toarray(), **rtol, atol=0, M=M, maxiter=maxiter) for i in range(B.shape[1])]
        if np.any([info > 0 for _, info in cg_out]):
            warn('Conjugate gradient convergence to tolerance not achieved. Consider decreasing beta to improve system conditionning.', stacklevel=2)
        X = np.asarray([x for x, _ in cg_out])
    return X