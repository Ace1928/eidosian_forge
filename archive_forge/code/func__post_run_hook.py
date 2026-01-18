from ..base import BaseInterface, BaseInterfaceInputSpec, traits
from ...utils.imagemanip import copy_header as _copy_header
def _post_run_hook(self, runtime):
    """Copy headers for outputs, if required."""
    runtime = super()._post_run_hook(runtime)
    if self._copy_header_map is None or not self.inputs.copy_header:
        return runtime
    inputs = self.inputs.get_traitsfree()
    outputs = self.aggregate_outputs(runtime=runtime).get_traitsfree()
    defined_outputs = set(outputs.keys()).intersection(self._copy_header_map.keys())
    for out in defined_outputs:
        inp = self._copy_header_map[out]
        keep_dtype = True
        if isinstance(inp, tuple):
            inp, keep_dtype = inp
        _copy_header(inputs[inp], outputs[out], keep_dtype=keep_dtype)
    return runtime