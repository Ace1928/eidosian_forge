import json
import logging
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torchgen.api.cpp as cpp
from torchgen.context import native_function_manager
from torchgen.model import (
from torchgen.static_runtime import config
class GenOpDispatcher:

    def out_variant(self, groups: Sequence[NativeFunctionsGroup], backend_index: BackendIndex) -> str:
        if not groups:
            return ''
        generated_type_variants = []
        for g in groups:
            with native_function_manager(g):
                assert is_supported(g)
                assert isinstance(g, NativeFunctionsGroup)
                generated_type_variant = self.out_variant_op_generator(g, backend_index)
                generated_type_variants.append(generated_type_variant)
        op_name = op_name_from_group(groups[0])
        body = '\n'.join(generated_type_variants)
        generated = f'\nREGISTER_OPERATOR_FUNCTOR(\n    aten::{op_name},\n    aten_{op_name},\n    [](Node* n) -> SROperator {{\n      {body}\n      LogAndDumpSchema(n);\n      return nullptr;\n    }});\n'
        return generated

    def view(self, groups: Sequence[NativeFunctionsViewGroup], backend_index: BackendIndex) -> str:
        if not groups:
            return ''
        generated_type_variants = []
        for g in groups:
            with native_function_manager(g):
                assert is_supported(g)
                assert isinstance(g, NativeFunctionsViewGroup)
                generated_type_variant = self.view_op_generator(g, backend_index)
                generated_type_variants.append(generated_type_variant)
        op_name = config.func_name_base_str(groups[0])
        body = '\n'.join(generated_type_variants)
        generated = f'\nREGISTER_NATIVE_OPERATOR_FUNCTOR(\n    aten::{op_name},\n    aten_{op_name},\n    [](Node* n) -> SROperator {{\n      {body}\n      LogAndDumpSchema(n);\n      return nullptr;\n    }});\n'
        return generated

    def out_variant_op_generator(self, g: NativeFunctionsGroup, backend_index: BackendIndex) -> str:
        functional = g.functional
        schema = str(functional.func)
        populated_argument = generate_arg_extraction(g.functional.func)
        functional_variant_call = generate_non_out_variant_call(g, backend_index)
        assert len(g.out.func.arguments.out) == 1
        out_variable_name = str(g.out.func.arguments.out[0].name)
        out_variant_call = generate_out_variant_call(g, backend_index)
        generated = f'\n      if (n->matches(torch::schema("aten::{schema}"))) {{\n        return [](ProcessedNode* p_node) {{\n          {populated_argument}\n          if (p_node->Output(0).isNone()) {{\n            p_node->Output(0) = {functional_variant_call};\n            return;\n          }}\n          auto& {out_variable_name} = p_node->Output(0).toTensor();\n          fastResizeToZero({out_variable_name});\n          {out_variant_call};\n        }};\n      }}'
        return generated

    def view_op_generator(self, g: NativeFunctionsViewGroup, backend_index: BackendIndex) -> str:
        schema = str(g.view.func)
        populated_argument = generate_arg_extraction(g.view.func)
        functional_variant_call = generate_call_to_view_ops(g, backend_index)
        generated = f'\n      if (n->matches(torch::schema("aten::{schema}"))) {{\n        return [](ProcessedNode* p_node) {{\n          {populated_argument}\n            p_node->Output(0) = {functional_variant_call};\n        }};\n      }}'
        return generated