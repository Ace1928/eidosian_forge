def _init_mat(nrows, ncols, ins_cost, del_cost):
    mat = np.empty((nrows, ncols))
    mat[0, :] = ins_cost * np.arange(ncols)
    mat[:, 0] = del_cost * np.arange(nrows)
    return mat