class DTypePromotionError(TypeError):
    """Multiple DTypes could not be converted to a common one.

    This exception derives from ``TypeError`` and is raised whenever dtypes
    cannot be converted to a single common one.  This can be because they
    are of a different category/class or incompatible instances of the same
    one (see Examples).

    Notes
    -----
    Many functions will use promotion to find the correct result and
    implementation.  For these functions the error will typically be chained
    with a more specific error indicating that no implementation was found
    for the input dtypes.

    Typically promotion should be considered "invalid" between the dtypes of
    two arrays when `arr1 == arr2` can safely return all ``False`` because the
    dtypes are fundamentally different.

    Examples
    --------
    Datetimes and complex numbers are incompatible classes and cannot be
    promoted:

    >>> np.result_type(np.dtype("M8[s]"), np.complex128)
    DTypePromotionError: The DType <class 'numpy.dtype[datetime64]'> could not
    be promoted by <class 'numpy.dtype[complex128]'>. This means that no common
    DType exists for the given inputs. For example they cannot be stored in a
    single array unless the dtype is `object`. The full list of DTypes is:
    (<class 'numpy.dtype[datetime64]'>, <class 'numpy.dtype[complex128]'>)

    For example for structured dtypes, the structure can mismatch and the
    same ``DTypePromotionError`` is given when two structured dtypes with
    a mismatch in their number of fields is given:

    >>> dtype1 = np.dtype([("field1", np.float64), ("field2", np.int64)])
    >>> dtype2 = np.dtype([("field1", np.float64)])
    >>> np.promote_types(dtype1, dtype2)
    DTypePromotionError: field names `('field1', 'field2')` and `('field1',)`
    mismatch.

    """
    pass