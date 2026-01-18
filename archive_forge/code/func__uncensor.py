import numpy as np
def _uncensor(self):
    """
        This function is used when a non-censored version of the data
        is needed to create a rough estimate of the parameters of a
        distribution via the method of moments or some similar method.
        The data is "uncensored" by taking the given endpoints as the
        data for the left- or right-censored data, and the mean for the
        interval-censored data.
        """
    data = np.concatenate((self._uncensored, self._left, self._right, self._interval.mean(axis=1)))
    return data