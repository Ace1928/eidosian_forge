from __future__ import annotations
import logging
import operator
import types
from typing import (
import torch
import torch._ops
import torch.fx
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import (
from onnxscript.function_libs.torch_lib import (  # type: ignore[import]
class _OnnxSchemaChecker:
    """
    The OnnxSchemaChecker class is a checker for ONNX OpSchema and param schema.

    It provides methods to check for input compatibility based on the OpSchema. It also
    provides a matching score to indicate how well the OpSchema matches the input and
    kwargs types. A function will be evaluated as perfect match, nearest match eligible,
    or no match.

    Here are some common examples in categories:

    1. [NOTE: Perfect match]: The number of inputs and attributes are exactly the same as
        the OpSchema. The types of inputs and attributes are exactly the same as the
        OpSchema.

        ```python
        inputs = (Tensor[2, 3], Tensor[2, 3])
        attributes = {"alpha": 1.0}

        @torch_op("aten::op")
        def aten_op(self: TReal, other: TReal, alpha: float = 1) -> TReal:
            ...

        ```
        Result: Perfect match.

    2. [NOTE: Optional input]: The dispatcher recognizes optional inputs. However,
        the input can't be ignored. None must be provided.

        ```python
        inputs = (Tensor([2, 3]), None)
        attributes = {}

        aten_op(X: TTensor, Y: Optional[INT64]):
            ...
        ```
        Result: Perfect match.
        Real example: `aten::convolution`.

    3. [NOTE: Different attributes]: If an attribute is provided with value, it's
        a must to match the attribute in function signature.
        ```python
        inputs = (Tensor([2, 3]),)
        attributes = {"a":1, "b":2}

        aten_op(X: TTensor, a: int):
            ...
        ```
        Result: No match.
        Real example: `aten::div` vs `aten::div.Tensor_mode`.

    4. [NOTE: Default attributes]: Default attribute will fill in the value into
        inputs/attributes.
        ```python
        inputs = (Tensor([2, 3]),)
        attributes = {}

        aten_op(X: TTensor, a: int = 3):
            ...
        ```
        Result: Perfect match.
        Real example: `aten::clone`

    5. [NOTE: Ignore attribute with None value]: The attributes with None value
        will be ignored in matching.
        ```python
        inputs = (Tensor([2, 3]),)
        attributes = {"a": None}

        aten_op(X: TTensor):
            ...
        ```
        Result: Perfect match.

        ```python
        inputs = (Tensor([2, 3]),)
        attributes = {"a": None}

        aten_op(X: TTensor, a: int = 3):
            ...
        ```
        Result: Nearest match eligible.

        Real example: `aten::div` vs `aten::div.Tensor_mode`.

    Attributes:
        onnxfunction: The OnnxFunction.
        param_schema: The parameter schema defined in the OnnxFunction.
        op_schema: The ONNX OpSchema.
        type_constraints: The type constraints defined in the OpSchema.
        attributes: The attributes defined in the OpSchema.
        _matching_score: The matching score of the OnnxSchemaChecker .

    """

    def __init__(self, onnxfunction: Union[onnxscript.OnnxFunction, onnxscript.TracedOnnxFunction]):
        """Initialize the OnnxSchemaChecker .

        Args:
            onnxfunction: The OnnxFunction.
        """
        self.onnxfunction = onnxfunction
        self.param_schema = self.onnxfunction.param_schemas()
        op_schema = self.onnxfunction.op_schema
        assert op_schema is not None
        self.op_schema = op_schema
        self.type_constraints = {constraint.type_param_str: set(constraint.allowed_type_strs) for constraint in self.op_schema.type_constraints}
        self.attributes = self.op_schema.attributes
        self._matching_score: Optional[int] = None

    @property
    def match_score(self) -> Optional[int]:
        """The matching score of the OnnxSchemaChecker .

        If this remains None, it means the matching score has not been calculated,
        and it's not a nearest match candidate.

        Returns:
            The matching score of the OnnxSchemaChecker .
        """
        return self._matching_score

    @_beartype.beartype
    def perfect_match_inputs(self, diagnostic: diagnostics.Diagnostic, args: Sequence[Optional[Union[fx_type_utils.TensorLike, str, int, float, bool, list]]], kwargs: Dict[str, fx_type_utils.Argument]) -> bool:
        """Check if the inputs perfectly match the OpSchema requirements.

        The definition of perfect match is that the input types are all in the type
        constraints and the number of inputs matches the number of inputs in the
        OpSchema.

        Checking steps:
        1. The function signature matches the inputs number, and attribute names.
        2. The input/attribute types are all in the type constraints.

        A function should at least pass the first step to be eligible for the
        nearest matching.

        Args:
            diagnostic: The diagnostic to use for logging detailed info.
            args: The input arguments organized in PyTorch inputs way.
            kwargs: The input keyword arguments organized in PyTorch inputs way.

        Returns:
            True if the inputs match the requirements, False otherwise.
        """
        function_inputs, function_attributes = self._separate_input_attributes_from_arguments(self.param_schema, args, kwargs, fill_defaults=True)
        with diagnostic.log_section(logging.INFO, 'Checking perfect match...'):
            diagnostic.info('%s', diagnostics.LazyString(diagnostics.format_argument, self.onnxfunction))
            is_perfect_match = True
            if len(function_inputs) != len(self.op_schema.inputs):
                with diagnostic.log_section(logging.INFO, 'Failed: input number mismatch!'):
                    diagnostic.info('Actual %d vs expected %d', len(function_inputs), len(self.op_schema.inputs))
                diagnostic.info('The function is not a nearest match candidate.')
                is_perfect_match = False
            if set(function_attributes) != set(self.attributes):
                with diagnostic.log_section(logging.INFO, 'Failed: attribute mismatch!'):
                    diagnostic.info('%s', diagnostics.LazyString(lambda: f'Actual {set(function_attributes)} vs expected {set(self.attributes)}'))
                diagnostic.info('The function is not a nearest match candidate.')
                is_perfect_match = False
            if not is_perfect_match:
                return False
            for schema_input, torch_input in zip(self.op_schema.inputs, function_inputs):
                torch_input_compatible_types = _find_onnx_data_type(torch_input)
                allowed_types = self.type_constraints[schema_input.type_str]
                if not allowed_types.intersection(torch_input_compatible_types) and (not any((fx_type_utils.is_optional_onnx_dtype_str(onnx_type_str) for onnx_type_str in allowed_types))):
                    with diagnostic.log_section(logging.INFO, "Failed: input type mismatch for input '%s'!", schema_input.name):
                        diagnostic.info('Actual %s vs\nExpected %s', torch_input_compatible_types, allowed_types)
                    is_perfect_match = False
            for attribute_name, attribute in function_attributes.items():
                if not self._match_onnx_attribute_type(attribute_name, attribute):
                    with diagnostic.log_section(logging.INFO, "Failed: attribute '%s' type mismatch!", attribute_name):
                        diagnostic.info('Actual %s vs\nExpected %s', type(attribute), self.attributes[attribute_name].type)
                    is_perfect_match = False
            self._record_matching_score(function_inputs, function_attributes)
            diagnostic.info('match score: %d', self.match_score)
            return is_perfect_match

    @_beartype.beartype
    def _match_onnx_attribute_type(self, attribute_name: str, attribute: Union[fx_type_utils.Argument, onnxscript_graph_building.TorchScriptTensor], is_sequence: bool=False) -> bool:
        if isinstance(attribute, (int, float, bool, str)):
            attribute_onnx_type = fx_type_utils.from_python_type_to_onnx_attribute_type(type(attribute), is_sequence=is_sequence)
            if attribute_onnx_type != self.attributes[attribute_name].type:
                return False
        elif isinstance(attribute, (list, tuple)) and attribute:
            return self._match_onnx_attribute_type(attribute_name, attribute[0], is_sequence=True)
        else:
            return False
        return True

    @_beartype.beartype
    def _record_matching_score(self, inputs: Sequence[Optional[Union[fx_type_utils.TensorLike, str, int, float, bool, list]]], attributes: Dict[str, fx_type_utils.Argument]):
        """Calculate the inputs matching score of the OpSchema requirements to find the nearest match.

        Only the functions which have the same number of inputs and attributes as the
        OpSchema are eligible to be a nearest match candidate. Thus, we don't need to
        check the length of inputs and attributes here, and only check the types of
        inputs and attributes.

        How the matchsing score is calculated:
            score += 1 if one input/attribute type is in the type constraints.

        Limitations:
            None/NoeType/[] could result in zero matches, and the same score of overloads,
            which will be recorded in SARIF.

        Args:
            inputs: The input arguments.
            attributes: The input keyword arguments.

        Returns:
            True if the inputs match the requirements, False otherwise.
        """
        self._matching_score = 0
        for schema_input, torch_input in zip(self.op_schema.inputs, inputs):
            torch_input_compatible_types = _find_onnx_data_type(torch_input)
            allowed_types = self.type_constraints[schema_input.type_str]
            if allowed_types.intersection(torch_input_compatible_types):
                self._matching_score += 1
        for attribute_name, attribute_proto in self.attributes.items():
            attribute = attributes[attribute_name]
            attribute_onnx_type = fx_type_utils.from_python_type_to_onnx_attribute_type(type(attribute))
            if attribute_onnx_type != attribute_proto.type:
                self._matching_score -= 1

    @_beartype.beartype
    def _separate_input_attributes_from_arguments(self, param_schemas: Sequence['onnxscript.values.ParamSchema'], args: Sequence[Optional[Union[fx_type_utils.TensorLike, str, int, float, bool, list]]], kwargs: Dict[str, fx_type_utils.Argument], fill_defaults: bool=True) -> Tuple[List[Any], Dict[str, Any]]:
        """Separate Python args and kwargs into ONNX inputs and attributes.

        Extra_kwargs are ignored if their values are None. For example, if the
        OpSchema has an attribute "rounding_mode" and the caller provides
        "rounding_mode=None", the attribute "rounding_mode" will not be included
        in the returned attributes when the OnnxFunction signature doesn't have
        "rounding_mode" as an attribute.

        Args:
            param_schemas: The parameter schemas of an Op or a OnnxFunction.
            args: The Python positional arguments supplied by the caller.
            kwargs: The Python keyword arguments supplied by the caller.
            fill_defaults: Whether to fill the default values for attributes.

        Returns:
            A tuple of two elements:
            - A list of ONNX inputs.
            - An dictionary of ONNX attribute names and values.

        Raises:
            TypeError: When allow_extra_kwargs is False and there are unknown kwargs.
            TypeError: When a required input is not provided.
        """
        import onnx
        onnx_inputs: List[Any] = []
        onnx_attributes: Dict[str, Any] = dict()
        copy_kwargs = kwargs.copy()
        for i, param in enumerate(param_schemas):
            if param.is_variadic_input:
                onnx_inputs.extend(args[i:])
                args = []
                continue
            if i < len(args):
                if param.is_input:
                    onnx_inputs.append(args[i])
                else:
                    onnx_attributes[param.name] = args[i]
            elif param.name in copy_kwargs:
                if param.is_input:
                    onnx_inputs.append(copy_kwargs[param.name])
                    copy_kwargs.pop(param.name)
                else:
                    onnx_attributes[param.name] = copy_kwargs[param.name]
            elif param.is_attribute and self.attributes[param.name].default_value.type != onnx.AttributeProto.UNDEFINED:
                if fill_defaults:
                    onnx_attributes[param.name] = param.default
            elif param.is_input:
                if fill_defaults:
                    onnx_inputs.append(None)
        for k, v in copy_kwargs.items():
            if k not in onnx_attributes and v is not None:
                onnx_attributes[k] = v
        return (onnx_inputs, onnx_attributes)