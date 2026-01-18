import itertools
from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import torchgen.api.dispatcher as dispatcher
from torchgen.api.lazy import (
from torchgen.api.translate import translate
from torchgen.api.types import (
from torchgen.context import method_with_native_function
from torchgen.dest.lazy_ts_lowering import ts_lowering_body
from torchgen.model import (
@dataclass(frozen=True)
class GenTSLazyIR(GenLazyIR):

    def lowering_function(self, schema: LazyIrSchema) -> str:
        signature = '\n  torch::lazy::TSOpVector Lower(\n      std::shared_ptr<torch::jit::GraphFunction> function,\n      torch::lazy::TSLoweringContext* loctx) const override'
        if schema.properties.LowerDeclOnly:
            return f'{signature};'
        elif schema.properties.Lower:
            return f'{signature} {{\n    {ts_lowering_body(schema)}\n  }}\n            '
        else:
            return ''

    def create_function(self, schema: LazyIrSchema, node_ctor_args: str) -> str:
        signature = f'static NodePtr Create({node_ctor_args})'
        if schema.properties.CreateFnDeclOnly:
            return f'{signature};'
        elif not schema.properties.CreateFn:
            return ''
        return f'{signature} {{\n    return ReuseOrMakeNode<{schema.node_name}>(data);\n  }}'

    def can_be_reused_function(self, schema: LazyIrSchema, node_ctor_args: str) -> str:
        signature = f'bool CanBeReused({node_ctor_args}) const'
        if schema.properties.CanBeReusedDeclOnly:
            return f'{signature};'
        elif not schema.properties.CanBeReused:
            return ''
        value_comparison = []
        for arg in itertools.chain(schema.positional_values, schema.keyword_values):
            if isinstance(arg.lazy_type, OptionalCType):
                value_comparison.append(f'nullable_operand(i++) == {arg.name}.value_or(kNullValue)')
            else:
                value_comparison.append(f'operand(i++) == {arg.name}')
        for arg in itertools.chain(schema.positional_scalars, schema.keyword_scalars):
            if isinstance(arg.lazy_type, OptionalCType):
                value_comparison.append(f'((!this->{arg.name}&&!{arg.name}) || (this->{arg.name}&&{arg.name} && *(this->{arg.name}) == *{arg.name}))')
            else:
                value_comparison.append(f'this->{arg.name} == {arg.name}')
        value_comparison_str = ' &&\n        '.join(value_comparison)
        return f'{signature} {{\n    size_t i = 0;\n    return ({value_comparison_str});\n  }}'