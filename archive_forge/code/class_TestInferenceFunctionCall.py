import unittest
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import onnx
from onnx import TensorProto, TypeProto
from onnx.checker import ValidationError
from onnx.defs import OpSchema, get_all_schemas_with_history, get_schema
from onnx.helper import (
from onnx.numpy_helper import from_array
from onnx.shape_inference import InferenceError, infer_node_outputs
class TestInferenceFunctionCall(unittest.TestCase):

    def test_add_inference(self) -> None:
        cases = [({'A': (TensorProto.FLOAT, ()), 'B': (TensorProto.FLOAT, ())}, {'C': (TensorProto.FLOAT, ())}), ({'A': (TensorProto.FLOAT, (None, 2)), 'B': (TensorProto.FLOAT, (2,))}, {'C': (TensorProto.FLOAT, (None, 2))}), ({'A': (TensorProto.FLOAT, (None, 2)), 'B': (TensorProto.FLOAT, (1, 2))}, {'C': (TensorProto.FLOAT, (None, 2))}), ({'A': (TensorProto.DOUBLE, ('n', 'm')), 'B': (TensorProto.DOUBLE, (1, 'n', 'm'))}, {'C': (TensorProto.DOUBLE, (1, 'n', 'm'))}), ({'A': (TensorProto.FLOAT, ('x', 2)), 'B': (TensorProto.FLOAT, ('y', 2))}, {'C': (TensorProto.FLOAT, (None, 2))})]
        for ins, outs in cases:
            assert _run_case(ADD_SCHEMA, ['A', 'B'], ['C'], _to_tensor_types(ins)) == _to_tensor_types(outs)

    def test_add_inference_raises_errors(self) -> None:
        with self.assertRaises(ValidationError):
            _run_case(ADD_SCHEMA, ['A'], ['C'], _to_tensor_types({'A': (TensorProto.FLOAT, (3, 4))}))
        with self.assertRaises(ValidationError):
            _run_case(ADD_SCHEMA, ['A', 'B'], ['C'], _to_tensor_types({'A': (TensorProto.FLOAT, (3, 4)), 'B': (2, (3, 4))}))
        with self.assertRaises(InferenceError):
            _run_case(ADD_SCHEMA, ['A', 'B'], ['C'], _to_tensor_types({'A': (TensorProto.FLOAT, (2, 4)), 'B': (TensorProto.FLOAT, (3, 4))}))
        with self.assertRaises(KeyError):
            _run_case(ADD_SCHEMA, ['A', 'B'], ['C'], _to_tensor_types({'A': (TensorProto.FLOAT, (3, 4))}))

    def test_reshape_inference(self) -> None:
        assert _run_case(RESHAPE_SCHEMA, ['x', 't'], ['y'], _to_tensor_types({'x': (TensorProto.FLOAT, (5, 4)), 't': (TensorProto.INT64, (3,))}), {'t': np.array([2, 2, 5], dtype=np.int64)}) == _to_tensor_types({'y': (TensorProto.FLOAT, (2, 2, 5))})

    def test_scan_inference_with_subgraph(self) -> None:
        seq_len = 'sequence'
        input_size = 2
        loop_state_size = 3
        input_value_infos = [make_tensor_value_info('loop_state_in', TensorProto.UNDEFINED, None), make_tensor_value_info('input', TensorProto.UNDEFINED, None), make_tensor_value_info('outer', TensorProto.UNDEFINED, None)]
        output_value_infos = [make_tensor_value_info('loop_state_out', TensorProto.UNDEFINED, None), make_tensor_value_info('output', TensorProto.FLOAT, (seq_len, input_size))]
        subgraph = make_graph([make_node('Identity', ['loop_state_in'], ['loop_state_out']), make_node('Add', ['input', 'outer'], ['output'])], 'subgraph', input_value_infos, output_value_infos)
        assert infer_node_outputs(get_schema('Scan', 9), make_node('Scan', ['loop_state_orig', 'scan_input', 'scan_outer'], ['loop_state_final', 'scan_output'], num_scan_inputs=1, body=subgraph), _to_tensor_types({'loop_state_orig': (TensorProto.FLOAT, (loop_state_size,)), 'scan_input': (TensorProto.FLOAT, (seq_len, input_size)), 'scan_outer': (TensorProto.FLOAT, (input_size,))}), opset_imports=[make_opsetid('', 9)], ir_version=4) == _to_tensor_types({'loop_state_final': (TensorProto.FLOAT, (loop_state_size,)), 'scan_output': (TensorProto.FLOAT, (seq_len, input_size))})

    def test_inference_with_conflow(self) -> None:
        model_script = '\n        <\n            ir_version: 8,\n            opset_import: ["" : 18, "onnxscript.atenlib" : 1],\n            producer_name: "pytorch",\n            producer_version: "2.1.0"\n        >\n        torch_jit (float input_0) => (float reault, int64 index)\n        {\n            reault, index = onnxscript.atenlib.aten_min_dim <dim = 0, keepdim = 1> (input_0)\n        }\n        <\n            domain: "onnxscript.atenlib",\n            opset_import: ["" : 18]\n        >\n        aten_min_dim <dim>(self) => (result_7, indices_6)\n        {\n            tmp = Shape (self)\n            tmp_0 = Size (tmp)\n            tmp_1 = Constant <value = int64 tmp_1 {0}> ()\n            tmp_1_cast = CastLike (tmp_1, tmp_0)\n            tmp_2 = Equal (tmp_0, tmp_1_cast)\n            cond = Not (tmp_2)\n            indices_6, result_7 = If (cond) <\n                then_branch = thenGraph_4 () => ( indices,  result) {\n                    dim = Constant <value_int: int = @dim> ()\n                    tmp_3 = Constant <value_ints = [-1]> ()\n                    dims = Reshape (dim, tmp_3)\n                    result = ReduceMin <keepdims: int = @keepdim> (self, dims)\n                    indices = ArgMin <axis: int = @dim, keepdims: int = @keepdim> (self)\n                }, else_branch = elseGraph_4 () => ( indices_4,  result_5) {\n                    indices_4 = Constant <value_int = 0> ()\n                    result_5 = Identity (self)\n                }\n            >\n        }\n        '
        model = onnx.parser.parse_model(model_script)
        onnx.shape_inference.infer_shapes(model, strict_mode=False)
        with self.assertRaises(onnx.shape_inference.InferenceError):
            onnx.shape_inference.infer_shapes(model, strict_mode=True)

    def test_inference_with_attribute(self) -> None:
        model_script = '\n        <\n            ir_version: 8,\n            opset_import: ["" : 18, "custom" : 1],\n            producer_name: "",\n            producer_version: "1.0"\n        >\n        MeanVarianceNormalization (float[N] x) => (float[M] y)\n        {\n            y = custom.custom_mvn <axes = [0]> (x)\n        }\n        <\n            domain: "custom",\n            opset_import: ["" : 18]\n        >\n        custom_mvn <axes>(X) => (Y)\n        {\n          Exponent = Constant <value = float {2.0}>()\n          Epsilon = Constant <value = float {1e-9}>()\n          axes = Constant <value_ints: ints = @axes>()\n          X_RM = ReduceMean (X, axes)\n          EX_squared = Pow (X_RM, Exponent)\n          X_squared = Pow (X, Exponent)\n          E_Xsquared = ReduceMean (X_squared, axes)\n          Variance = Sub (E_Xsquared, EX_squared)\n          STD = Sqrt (Variance)\n          X_variance = Sub (X, X_RM)\n          Processed_STD = Add (STD, Epsilon)\n          Y = Div (X_variance, Processed_STD)\n        }\n        '
        model = onnx.parser.parse_model(model_script)
        onnx.shape_inference.infer_shapes(model, strict_mode=True)