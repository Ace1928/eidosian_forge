import numpy as np
from matplotlib import _api
from matplotlib.tri import Triangulation
from matplotlib.tri._trifinder import TriFinder
from matplotlib.tri._tritools import TriAnalyzer
class _DOF_estimator_geom(_DOF_estimator):
    """Fast 'geometric' approximation, recommended for large arrays."""

    def compute_dz(self):
        """
        self.df is computed as weighted average of _triangles sharing a common
        node. On each triangle itri f is first assumed linear (= ~f), which
        allows to compute d~f[itri]
        Then the following approximation of df nodal values is then proposed:
            f[ipt] = SUM ( w[itri] x d~f[itri] , for itri sharing apex ipt)
        The weighted coeff. w[itri] are proportional to the angle of the
        triangle itri at apex ipt
        """
        el_geom_w = self.compute_geom_weights()
        el_geom_grad = self.compute_geom_grads()
        w_node_sum = np.bincount(np.ravel(self._triangles), weights=np.ravel(el_geom_w))
        dfx_el_w = np.empty_like(el_geom_w)
        dfy_el_w = np.empty_like(el_geom_w)
        for iapex in range(3):
            dfx_el_w[:, iapex] = el_geom_w[:, iapex] * el_geom_grad[:, 0]
            dfy_el_w[:, iapex] = el_geom_w[:, iapex] * el_geom_grad[:, 1]
        dfx_node_sum = np.bincount(np.ravel(self._triangles), weights=np.ravel(dfx_el_w))
        dfy_node_sum = np.bincount(np.ravel(self._triangles), weights=np.ravel(dfy_el_w))
        dfx_estim = dfx_node_sum / w_node_sum
        dfy_estim = dfy_node_sum / w_node_sum
        return np.vstack([dfx_estim, dfy_estim]).T

    def compute_geom_weights(self):
        """
        Build the (nelems, 3) weights coeffs of _triangles angles,
        renormalized so that np.sum(weights, axis=1) == np.ones(nelems)
        """
        weights = np.zeros([np.size(self._triangles, 0), 3])
        tris_pts = self._tris_pts
        for ipt in range(3):
            p0 = tris_pts[:, ipt % 3, :]
            p1 = tris_pts[:, (ipt + 1) % 3, :]
            p2 = tris_pts[:, (ipt - 1) % 3, :]
            alpha1 = np.arctan2(p1[:, 1] - p0[:, 1], p1[:, 0] - p0[:, 0])
            alpha2 = np.arctan2(p2[:, 1] - p0[:, 1], p2[:, 0] - p0[:, 0])
            angle = np.abs((alpha2 - alpha1) / np.pi % 1)
            weights[:, ipt] = 0.5 - np.abs(angle - 0.5)
        return weights

    def compute_geom_grads(self):
        """
        Compute the (global) gradient component of f assumed linear (~f).
        returns array df of shape (nelems, 2)
        df[ielem].dM[ielem] = dz[ielem] i.e. df = dz x dM = dM.T^-1 x dz
        """
        tris_pts = self._tris_pts
        tris_f = self.z[self._triangles]
        dM1 = tris_pts[:, 1, :] - tris_pts[:, 0, :]
        dM2 = tris_pts[:, 2, :] - tris_pts[:, 0, :]
        dM = np.dstack([dM1, dM2])
        dM_inv = _safe_inv22_vectorized(dM)
        dZ1 = tris_f[:, 1] - tris_f[:, 0]
        dZ2 = tris_f[:, 2] - tris_f[:, 0]
        dZ = np.vstack([dZ1, dZ2]).T
        df = np.empty_like(dZ)
        df[:, 0] = dZ[:, 0] * dM_inv[:, 0, 0] + dZ[:, 1] * dM_inv[:, 1, 0]
        df[:, 1] = dZ[:, 0] * dM_inv[:, 0, 1] + dZ[:, 1] * dM_inv[:, 1, 1]
        return df