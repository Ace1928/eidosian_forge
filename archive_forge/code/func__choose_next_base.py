def _choose_next_base(self, base_tree_remaining):
    """
        Return the next base.

        The return value will either fit the C3 constraints or be our best
        guess about what to do. If we cannot guess, this may raise an exception.
        """
    base = self._find_next_C3_base(base_tree_remaining)
    if base is not None:
        return base
    return self._guess_next_base(base_tree_remaining)