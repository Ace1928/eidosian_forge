import ast
import re
import sys
import warnings
from ..exceptions import DTypePromotionError
from .multiarray import dtype, array, ndarray, promote_types
def _promote_fields(dt1, dt2):
    """ Perform type promotion for two structured dtypes.

    Parameters
    ----------
    dt1 : structured dtype
        First dtype.
    dt2 : structured dtype
        Second dtype.

    Returns
    -------
    out : dtype
        The promoted dtype

    Notes
    -----
    If one of the inputs is aligned, the result will be.  The titles of
    both descriptors must match (point to the same field).
    """
    if (dt1.names is None or dt2.names is None) or dt1.names != dt2.names:
        raise DTypePromotionError(f'field names `{dt1.names}` and `{dt2.names}` mismatch.')
    identical = dt1 is dt2
    new_fields = []
    for name in dt1.names:
        field1 = dt1.fields[name]
        field2 = dt2.fields[name]
        new_descr = promote_types(field1[0], field2[0])
        identical = identical and new_descr is field1[0]
        if field1[2:] != field2[2:]:
            raise DTypePromotionError(f"field titles of field '{name}' mismatch")
        if len(field1) == 2:
            new_fields.append((name, new_descr))
        else:
            new_fields.append(((field1[2], name), new_descr))
    res = dtype(new_fields, align=dt1.isalignedstruct or dt2.isalignedstruct)
    if identical and res.itemsize == dt1.itemsize:
        for name in dt1.names:
            if dt1.fields[name][1] != res.fields[name][1]:
                return res
        return dt1
    return res