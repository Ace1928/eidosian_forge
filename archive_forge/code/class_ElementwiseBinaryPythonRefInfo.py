from torch.testing._internal.opinfo.core import (
class ElementwiseBinaryPythonRefInfo(BinaryUfuncInfo):
    """
    An OpInfo for a Python reference of an elementwise binary operation.
    """

    def __init__(self, name, *, op=None, op_db=None, torch_opinfo_name, torch_opinfo_variant_name='', **kwargs):
        self.torch_opinfo_name = torch_opinfo_name
        self.torch_opinfo_variant_name = torch_opinfo_variant_name
        self.torch_opinfo = _find_referenced_opinfo(torch_opinfo_name, torch_opinfo_variant_name, op_db=op_db)
        assert isinstance(self.torch_opinfo, BinaryUfuncInfo)
        inherited = self.torch_opinfo._original_binary_ufunc_args
        ukwargs = _inherit_constructor_args(name, op, inherited, kwargs)
        super().__init__(**ukwargs)