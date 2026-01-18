import numpy as np
from scipy import sparse, stats
from scipy.sparse import linalg
from pygsp import graphs, filters, utils
def _pyramid_single_interpolation(G, ca, pe, keep_inds, h_filter, **kwargs):
    """Synthesize a single level of the graph pyramid transform.

    Parameters
    ----------
    G : Graph
        Graph structure on which the signal resides.
    ca : ndarray
        Coarse approximation of the signal on a reduced graph.
    pe : ndarray
        Prediction error that was made when forming the current coarse approximation.
    keep_inds : ndarray
        The indices of the vertices to keep when downsampling the graph and signal.
    h_filter : lambda expression
        The filter in use at this level.
    use_landweber : bool
        To use the Landweber iteration approximation in the least squares synthesis.
        Default is False.
    reg_eps : float
        Interpolation parameter. Default is 0.005.
    landweber_its : int
        Number of iterations in the Landweber approximation for least squares synthesis.
        Default is 50.
    landweber_tau : float
        Parameter for the Landweber iteration. Default is 1.

    Returns
    -------
    finer_approx :
        Coarse approximation of the signal on a higher resolution graph.

    """
    nb_ind = keep_inds.shape
    N = G.N
    reg_eps = float(kwargs.pop('reg_eps', 0.005))
    use_landweber = bool(kwargs.pop('use_landweber', False))
    landweber_its = int(kwargs.pop('landweber_its', 50))
    landweber_tau = float(kwargs.pop('landweber_tau', 1.0))
    S = sparse.csr_matrix(([1] * nb_ind, (range(nb_ind), keep_inds)), shape=(nb_ind, N))
    if use_landweber:
        x = np.zeros(N)
        z = np.concatenate((ca, pe), axis=0)
        green_kernel = filters.Filter(G, lambda x: 1.0 / (x + reg_eps))
        PhiVlt = _analysis(green_kernel, S.T, **kwargs).T
        filt = filters.Filter(G, h_filter, **kwargs)
        for iteration in range(landweber_its):
            h_filtered_sig = _analysis(filt, x, **kwargs)
            x_bar = h_filtered_sig[keep_inds]
            y_bar = x - interpolate(G, x_bar, keep_inds, **kwargs)
            z_delt = np.concatenate((x_bar, y_bar), axis=0)
            z_delt = z - z_delt
            alpha_new = PhiVlt * z_delt[nb_ind:]
            x_up = sparse.csr_matrix((z_delt, (range(nb_ind), [1] * nb_ind)), shape=(N, 1))
            reg_L = G.L + reg_esp * sparse.eye(N)
            elim_inds = np.setdiff1d(np.arange(N, dtype=int), keep_inds)
            L_red = reg_L[np.ix_(keep_inds, keep_inds)]
            L_in_out = reg_L[np.ix_(keep_inds, elim_inds)]
            L_out_in = reg_L[np.ix_(elim_inds, keep_inds)]
            L_comp = reg_L[np.ix_(elim_inds, elim_inds)]
            next_term = L_red * alpha_new - L_in_out * linalg.spsolve(L_comp, L_out_in * alpha_new)
            next_up = sparse.csr_matrix((next_term, (keep_inds, [1] * nb_ind)), shape=(N, 1))
            x += landweber_tau * _analysis(filt, x_up - next_up, **kwargs) + z_delt[nb_ind:]
        finer_approx = x
    else:
        H = G.U * sparse.diags(h_filter(G.e), 0) * G.U.T
        Phi = G.U * sparse.diags(1.0 / (reg_eps + G.e), 0) * G.U.T
        Ta = np.concatenate((S * H, sparse.eye(G.N) - Phi[:, keep_inds] * linalg.spsolve(Phi[np.ix_(keep_inds, keep_inds)], S * H)), axis=0)
        finer_approx = linalg.spsolve(Ta.T * Ta, Ta.T * np.concatenate((ca, pe), axis=0))