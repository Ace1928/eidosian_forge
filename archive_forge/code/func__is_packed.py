import numpy as np
def _is_packed(dtype):
    """
    Checks whether the structured data type in 'dtype'
    has a simple layout, where all the fields are in order,
    and follow each other with no alignment padding.

    When this returns true, the dtype can be reconstructed
    from a list of the field names and dtypes with no additional
    dtype parameters.

    Duplicates the C `is_dtype_struct_simple_unaligned_layout` function.
    """
    align = dtype.isalignedstruct
    max_alignment = 1
    total_offset = 0
    for name in dtype.names:
        fld_dtype, fld_offset, title = _unpack_field(*dtype.fields[name])
        if align:
            total_offset = _aligned_offset(total_offset, fld_dtype.alignment)
            max_alignment = max(max_alignment, fld_dtype.alignment)
        if fld_offset != total_offset:
            return False
        total_offset += fld_dtype.itemsize
    if align:
        total_offset = _aligned_offset(total_offset, max_alignment)
    if total_offset != dtype.itemsize:
        return False
    return True