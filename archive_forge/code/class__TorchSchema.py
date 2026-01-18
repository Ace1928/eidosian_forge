import inspect
from typing import Dict, List, Union
from torch import _C
from torch.onnx import _constants
from torch.onnx._internal import registration
class _TorchSchema:

    def __init__(self, schema: Union[_C.FunctionSchema, str]) -> None:
        if isinstance(schema, _C.FunctionSchema):
            self.name: str = schema.name
            self.overload_name: str = schema.overload_name
            self.arguments: List[str] = [arg.name for arg in schema.arguments]
            self.optional_arguments: List[str] = []
            self.returns: List[str] = [ret.name for ret in schema.returns]
            self.opsets: List[int] = []
        else:
            self.name = schema
            self.overload_name = ''
            self.arguments = []
            self.optional_arguments = []
            self.returns = []
            self.opsets = []

    def __str__(self) -> str:
        s = f'{self.name}.{self.overload_name}(' + ', '.join(self.arguments) + ') -> (' + ', '.join(self.returns) + ')' + ' in opsets ' + ', '.join((str(opset) for opset in self.opsets))
        return s

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other) -> bool:
        if not isinstance(other, _TorchSchema):
            return False
        return self.name == other.name

    def is_aten(self) -> bool:
        return self.name.startswith('aten::')

    def is_backward(self) -> bool:
        return 'backward' in self.name