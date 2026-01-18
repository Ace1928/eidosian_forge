from __future__ import annotations
import textwrap
from typing import Optional
from torch import _C
from torch.onnx import _constants
from torch.onnx._internal import diagnostics
class SymbolicValueError(OnnxExporterError):
    """Errors around TorchScript values and nodes."""

    def __init__(self, msg: str, value: _C.Value):
        message = f"{msg}  [Caused by the value '{value}' (type '{value.type()}') in the TorchScript graph. The containing node has kind '{value.node().kind()}'.] "
        code_location = value.node().sourceRange()
        if code_location:
            message += f'\n    (node defined in {code_location})'
        try:
            message += '\n\n'
            message += textwrap.indent('Inputs:\n' + ('\n'.join((f"    #{i}: {input_}  (type '{input_.type()}')" for i, input_ in enumerate(value.node().inputs()))) or '    Empty') + '\n' + 'Outputs:\n' + ('\n'.join((f"    #{i}: {output}  (type '{output.type()}')" for i, output in enumerate(value.node().outputs()))) or '    Empty'), '    ')
        except AttributeError:
            message += ' Failed to obtain its input and output for debugging. Please refer to the TorchScript graph for debugging information.'
        super().__init__(message)