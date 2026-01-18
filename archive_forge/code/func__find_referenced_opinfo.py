from torch.testing._internal.opinfo.core import (
def _find_referenced_opinfo(referenced_name, variant_name, *, op_db=None):
    """
    Finds the OpInfo with the given name that has no variant name.
    """
    if op_db is None:
        from torch.testing._internal.common_methods_invocations import op_db
    for opinfo in op_db:
        if opinfo.name == referenced_name and opinfo.variant_test_name == variant_name:
            return opinfo