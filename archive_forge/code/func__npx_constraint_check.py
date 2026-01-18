def _npx_constraint_check(x, msg):
    """
    This operator will check if all the elements in a boolean tensor is true.
    If not, ValueError exception will be raised in the backend with given error message.
    In order to evaluate this operator, one should multiply the origin tensor by the return value
    of this operator to force this operator become part of the computation graph,
    otherwise the check would not be working under symoblic mode.

    Parameters
    ----------
    x : ndarray
        A boolean tensor.
    msg : string
        The error message in the exception.

    Returns
    -------
    out : ndarray
        If all the elements in the input tensor are true,
        array(True) will be returned, otherwise ValueError exception would
        be raised before anything got returned.

    Examples
    --------
    >>> loc = np.zeros((2,2))
    >>> scale = np.array(#some_value)
    >>> constraint = (scale > 0)
    >>> np.random.normal(loc,
                     scale * npx.constraint_check(constraint, 'Scale should be larger than zero'))

    If elements in the scale tensor are all bigger than zero, npx.constraint_check would return
    `np.array(True)`, which will not change the value of `scale` when multiplied by.
    If some of the elements in the scale tensor violate the constraint,
    i.e. there exists `False` in the boolean tensor `constraint`,
    a `ValueError` exception with given message 'Scale should be larger than zero' would be raised.
    """
    pass