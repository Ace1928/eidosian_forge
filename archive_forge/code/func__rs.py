import warnings
def _rs(self, results):
    """Normalize a list of results to a Resultset.

        A ResultSet is more consistent with the rest of Beautiful
        Soup's API, and ResultSet.__getattr__ has a helpful error
        message if you try to treat a list of results as a single
        result (a common mistake).
        """
    from bs4.element import ResultSet
    return ResultSet(None, results)