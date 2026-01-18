from torch.testing._internal.opinfo.core import (
class PythonRefInfo(OpInfo):
    """
    An OpInfo for a Python reference of an OpInfo base class operation.
    """

    def __init__(self, name, *, op=None, op_db=None, torch_opinfo_name, torch_opinfo_variant_name='', validate_view_consistency=True, **kwargs):
        self.torch_opinfo_name = torch_opinfo_name
        self.torch_opinfo_variant_name = torch_opinfo_variant_name
        self.torch_opinfo = _find_referenced_opinfo(torch_opinfo_name, torch_opinfo_variant_name, op_db=op_db)
        self.validate_view_consistency = validate_view_consistency
        assert isinstance(self.torch_opinfo, OpInfo)
        inherited = self.torch_opinfo._original_opinfo_args
        ukwargs = _inherit_constructor_args(name, op, inherited, kwargs)
        super().__init__(**ukwargs)