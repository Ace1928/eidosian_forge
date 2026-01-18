import numpy
Testing Goodness-of-fit Test with Pearson's Chi-squared Test.

    Args:
        observed (list of ints): List of # of counts each element is observed.
        expected (list of floats): List of # of counts each element is expected
            to be observed.
        alpha (float): Significance level. Currently,
            only 0.05 and 0.01 are acceptable.
        df (int): Degree of freedom. If ``None``,
            it is set to the length of ``observed`` minus 1.

    Returns:
        bool: ``True`` if null hypothesis is **NOT** reject.
        Otherwise, ``False``.
    