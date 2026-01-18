import networkx as nx
from networkx.utils import np_random_state
def _kamada_kawai_costfn(pos_vec, np, invdist, meanweight, dim):
    nNodes = invdist.shape[0]
    pos_arr = pos_vec.reshape((nNodes, dim))
    delta = pos_arr[:, np.newaxis, :] - pos_arr[np.newaxis, :, :]
    nodesep = np.linalg.norm(delta, axis=-1)
    direction = np.einsum('ijk,ij->ijk', delta, 1 / (nodesep + np.eye(nNodes) * 0.001))
    offset = nodesep * invdist - 1.0
    offset[np.diag_indices(nNodes)] = 0
    cost = 0.5 * np.sum(offset ** 2)
    grad = np.einsum('ij,ij,ijk->ik', invdist, offset, direction) - np.einsum('ij,ij,ijk->jk', invdist, offset, direction)
    sumpos = np.sum(pos_arr, axis=0)
    cost += 0.5 * meanweight * np.sum(sumpos ** 2)
    grad += meanweight * sumpos
    return (cost, grad.ravel())