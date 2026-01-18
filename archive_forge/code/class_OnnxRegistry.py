from __future__ import (  # for onnx.ModelProto (ONNXProgram) and onnxruntime (ONNXRuntimeOptions)
import abc
import contextlib
import dataclasses
import io
import logging
import os
import warnings
from collections import defaultdict
from typing import (
from typing_extensions import Self
import torch
import torch._ops
import torch.export as torch_export
import torch.utils._pytree as pytree
from torch._subclasses import fake_tensor
from torch.onnx._internal import _beartype, io_adapter
from torch.onnx._internal.diagnostics import infra
from torch.onnx._internal.fx import (
class OnnxRegistry:
    """Registry for ONNX functions.

    The registry maintains a mapping from qualified names to symbolic functions under a
    fixed opset version. It supports registering custom onnx-script functions and for
    dispatcher to dispatch calls to the appropriate function.

    """

    def __init__(self) -> None:
        """Initializes the registry"""
        self._registry: Dict[registration.OpName, List[registration.ONNXFunction]] = defaultdict(list)
        from onnxscript.function_libs.torch_lib import ops, registration
        self._opset_version = _DEFAULT_OPSET_VERSION
        warnings.warn(f'torch.onnx.dynamo_export only implements opset version {self._opset_version} for now. If you need to use a different opset version, please register them with register_custom_op.')
        self._initiate_registry_from_torchlib(registration.default_registry)

    @property
    def opset_version(self) -> int:
        """The ONNX opset version the exporter should target. Defaults to the latest
        supported ONNX opset version: 18. The default version will increment over time as
        ONNX continues to evolve."""
        return self._opset_version

    def _initiate_registry_from_torchlib(self, torchlib_registry: torchlib_registry.Registry):
        """Populates the registry with ATen functions from torchlib.

        Args:
            torchlib_registry: The torchlib registry to use for populating the registry.
        """
        for aten_name, aten_overloads_func in torchlib_registry.items():
            internal_name_instance = registration.OpName.from_qualified_name(aten_name)
            for overload_func in aten_overloads_func.overloads:
                symbolic_function = registration.ONNXFunction(onnx_function=overload_func, op_full_name=internal_name_instance.qualified_name(), is_custom=False, is_complex=False)
                self._register(internal_name_instance, symbolic_function)
            for complex_func in aten_overloads_func.complex:
                symbolic_function = registration.ONNXFunction(onnx_function=complex_func, op_full_name=internal_name_instance.qualified_name(), is_custom=False, is_complex=True)
                self._register(internal_name_instance, symbolic_function)

    @_beartype.beartype
    def _register(self, internal_qualified_name: registration.OpName, symbolic_function: registration.ONNXFunction) -> None:
        """Registers a ONNXFunction to an operator.

        Args:
            internal_qualified_name: The qualified name of the operator to register: OpName.
            symbolic_function: The ONNXFunction to register.
        """
        self._registry[internal_qualified_name].append(symbolic_function)

    @_beartype.beartype
    def register_op(self, function: Union['onnxscript.OnnxFunction', 'onnxscript.TracedOnnxFunction'], namespace: str, op_name: str, overload: Optional[str]=None, is_complex: bool=False) -> None:
        """Registers a custom operator: torch.ops.<namespace>.<op_name>.<overload>.

        Args:
            function: The onnx-sctip function to register.
            namespace: The namespace of the operator to register.
            op_name: The name of the operator to register.
            overload: The overload of the operator to register. If it's default overload,
                leave it to None.
            is_complex: Whether the function is a function that handles complex valued inputs.

        Raises:
            ValueError: If the name is not in the form of 'namespace::op'.
        """
        internal_name_instance = registration.OpName.from_name_parts(namespace=namespace, op_name=op_name, overload=overload)
        symbolic_function = registration.ONNXFunction(onnx_function=function, op_full_name=internal_name_instance.qualified_name(), is_custom=True, is_complex=is_complex)
        self._register(internal_name_instance, symbolic_function)

    @_beartype.beartype
    def get_op_functions(self, namespace: str, op_name: str, overload: Optional[str]=None) -> Optional[List[registration.ONNXFunction]]:
        """Returns a list of ONNXFunctions for the given op: torch.ops.<namespace>.<op_name>.<overload>.

        The list is ordered by the time of registration. The custom operators should be
        in the second half of the list.

        Args:
            namespace: The namespace of the operator to get.
            op_name: The name of the operator to get.
            overload: The overload of the operator to get. If it's default overload,
                leave it to None.
        Returns:
            A list of ONNXFunctions corresponding to the given name, or None if
            the name is not in the registry.
        """
        internal_name_instance = registration.OpName.from_name_parts(namespace=namespace, op_name=op_name, overload=overload)
        return self._registry.get(internal_name_instance)

    @_beartype.beartype
    def is_registered_op(self, namespace: str, op_name: str, overload: Optional[str]=None) -> bool:
        """Returns whether the given op is registered: torch.ops.<namespace>.<op_name>.<overload>.

        Args:
            namespace: The namespace of the operator to check.
            op_name: The name of the operator to check.
            overload: The overload of the operator to check. If it's default overload,
                leave it to None.

        Returns:
            True if the given op is registered, otherwise False.
        """
        functions = self.get_op_functions(namespace=namespace, op_name=op_name, overload=overload)
        return functions is not None

    @_beartype.beartype
    def _all_registered_ops(self) -> Set[str]:
        """Returns the set of all registered function names."""
        return {op_name_class.qualified_name() for op_name_class in self._registry.keys()}