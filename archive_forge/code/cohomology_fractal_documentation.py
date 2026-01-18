from snappy.ptolemy.homology import homology_basis_representatives_with_orders

    Convenience method to quickly access cohomology classes for
    M.inside_view().

        >>> from snappy import Manifold
        >>> compute_weights_basis_class(Manifold("m004"), None)
        (None, None, None)
        >>> compute_weights_basis_class(Manifold("m003"), [1, 0, 0, 1, -1, 0, 0, -1])
        ([1, 0, 0, 1, -1, 0, 0, -1], None, None)
        >>> compute_weights_basis_class(Manifold("m003"), 0)
        (None, [[1, 0, 0, 1, -1, 0, 0, -1]], [1.0])
        >>> compute_weights_basis_class(Manifold("m125"), 0)
        (None, [[-1, -4, -2, 0, 1, 0, 0, 0, 0, 4, 1, 2, -1, 0, 0, 0], [0, 5, 2, 0, 0, 0, 0, 1, 0, -5, 0, -2, 0, 0, 0, -1]], [1.0, 0.0])
        >>> compute_weights_basis_class(Manifold("m125"), 1)
        (None, [[-1, -4, -2, 0, 1, 0, 0, 0, 0, 4, 1, 2, -1, 0, 0, 0], [0, 5, 2, 0, 0, 0, 0, 1, 0, -5, 0, -2, 0, 0, 0, -1]], [0.0, 1.0])
        >>> compute_weights_basis_class(Manifold("m125"), [0.5, 0.5])
        (None, [[-1, -4, -2, 0, 1, 0, 0, 0, 0, 4, 1, 2, -1, 0, 0, 0], [0, 5, 2, 0, 0, 0, 0, 1, 0, -5, 0, -2, 0, 0, 0, -1]], [0.5, 0.5])

    