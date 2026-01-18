import os
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datasets import Dataset
from packaging.version import Version, parse
from onnxruntime import __version__ as ort_version
from onnxruntime.quantization import CalibraterBase, CalibrationMethod, QuantFormat, QuantizationMode, QuantType
from onnxruntime.quantization.calibrate import create_calibrator
from onnxruntime.quantization.registry import IntegerOpsRegistry, QDQRegistry, QLinearOpsRegistry
from onnxruntime.transformers.fusion_options import FusionOptions
from ..configuration_utils import BaseConfig
from ..utils import logging
@dataclass
class OptimizationConfig:
    """
    OptimizationConfig is the configuration class handling all the ONNX Runtime optimization parameters.
    There are two stacks of optimizations:
        1. The ONNX Runtime general-purpose optimization tool: it can work on any ONNX model.
        2. The ONNX Runtime transformers optimization tool: it can only work on a subset of transformers models.

    Attributes:
        optimization_level (`int`, defaults to 1):
            Optimization level performed by ONNX Runtime of the loaded graph.
            Supported optimization level are 0, 1, 2 and 99.
                - 0: will disable all optimizations
                - 1: will enable basic optimizations
                - 2: will enable basic and extended optimizations, including complex node fusions applied to the nodes
                assigned to the CPU or CUDA execution provider, making the resulting optimized graph hardware dependent
                - 99: will enable all available optimizations including layout optimizations
        optimize_for_gpu (`bool`, defaults to `False`):
            Whether to optimize the model for GPU inference.
            The optimized graph might contain operators for GPU or CPU only when `optimization_level` > 1.
        fp16 (`bool`, defaults to `False`):
            Whether all weights and nodes should be converted from float32 to float16.
        enable_transformers_specific_optimizations (`bool`, defaults to `True`):
            Whether to only use `transformers` specific optimizations on top of ONNX Runtime general optimizations.
        disable_gelu_fusion (`bool`, defaults to `False`):
            Whether to disable the Gelu fusion.
        disable_layer_norm_fusion (`bool`, defaults to `False`):
            Whether to disable Layer Normalization fusion.
        disable_attention_fusion (`bool`, defaults to `False`):
            Whether to disable Attention fusion.
        disable_skip_layer_norm_fusion (`bool`, defaults to `False`):
            Whether to disable SkipLayerNormalization fusion.
        disable_bias_skip_layer_norm_fusion (`bool`, defaults to `False`):
            Whether to disable Add Bias and SkipLayerNormalization fusion.
        disable_bias_gelu_fusion (`bool`, defaults to `False`):
            Whether to disable Add Bias and Gelu / FastGelu fusion.
        disable_embed_layer_norm_fusion (`bool`, defaults to `True`):
            Whether to disable EmbedLayerNormalization fusion.
            The default value is set to `True` since this fusion is incompatible with ONNX Runtime quantization.
        enable_gelu_approximation (`bool`, defaults to `False`):
            Whether to enable Gelu / BiasGelu to FastGelu conversion.
            The default value is set to `False` since this approximation might slightly impact the model's accuracy.
        use_mask_index (`bool`, defaults to `False`):
            Whether to use mask index instead of raw attention mask in the attention operator.
        no_attention_mask (`bool`, defaults to `False`):
            Whether to not use attention masks. Only works for bert model type.
        disable_embed_layer_norm (`bool`, defaults to `True`):
            Whether to disable EmbedLayerNormalization fusion.
            The default value is set to `True` since this fusion is incompatible with ONNX Runtime quantization
        disable_shape_inference (`bool`, defaults to `False`):
            Whether to disable symbolic shape inference.
            The default value is set to `False` but symbolic shape inference might cause issues sometimes.
        use_multi_head_attention (`bool`, defaults to `False`):
            Experimental argument. Use MultiHeadAttention instead of Attention operator, which has merged weights for Q/K/V projection,
            which might be faster in some cases since 3 MatMul is merged into one."
            "Note that MultiHeadAttention might be slower than Attention when qkv are not packed. "
        enable_gemm_fast_gelu_fusion (`bool`, defaults to `False`):
            Enable GemmfastGelu fusion.
        use_raw_attention_mask (`bool`, defaults to `False`):
            Use raw attention mask. Use this option if your input is not right-side padding. This might deactivate fused attention and get worse performance.
        disable_group_norm_fusion (`bool`, defaults to `True`):
            Do not fuse GroupNorm. Only works for model_type=unet.
        disable_packed_kv (`bool`, defaults to `True`):
            Do not use packed kv in cross attention. Only works for model_type=unet.
        disable_rotary_embeddings (`bool`, defaults to `False`):
            Whether to disable Rotary Embedding fusion.
    """
    optimization_level: int = 1
    optimize_for_gpu: bool = False
    fp16: bool = False
    optimize_with_onnxruntime_only: Optional[bool] = None
    enable_transformers_specific_optimizations: bool = True
    disable_gelu: Optional[bool] = None
    disable_gelu_fusion: bool = False
    disable_layer_norm: Optional[bool] = None
    disable_layer_norm_fusion: bool = False
    disable_attention: Optional[bool] = None
    disable_attention_fusion: bool = False
    disable_skip_layer_norm: Optional[bool] = None
    disable_skip_layer_norm_fusion: bool = False
    disable_bias_skip_layer_norm: Optional[bool] = None
    disable_bias_skip_layer_norm_fusion: bool = False
    disable_bias_gelu: Optional[bool] = None
    disable_bias_gelu_fusion: bool = False
    disable_embed_layer_norm: Optional[bool] = None
    disable_embed_layer_norm_fusion: bool = True
    enable_gelu_approximation: bool = False
    use_mask_index: bool = False
    no_attention_mask: bool = False
    disable_embed_layer_norm: bool = True
    disable_shape_inference: bool = False
    use_multi_head_attention: bool = False
    enable_gemm_fast_gelu_fusion: bool = False
    use_raw_attention_mask: bool = False
    disable_group_norm_fusion: bool = True
    disable_packed_kv: bool = True
    disable_rotary_embeddings: bool = False

    def __post_init__(self):

        def deprecate_renamed_attribute(old_name, new_name, mapping_func=None):
            if getattr(self, old_name, None) is not None:
                if mapping_func is None:

                    def identity(x):
                        return x
                    mapping_func = identity
                setattr(self, new_name, mapping_func(getattr(self, old_name)))
                warnings.warn(f'{old_name} will be deprecated soon, use {new_name} instead, {new_name} is set to {getattr(self, new_name)}.', FutureWarning)
        deprecate_renamed_attribute('optimize_with_onnxruntime_only', 'enable_transformers_specific_optimizations', mapping_func=lambda x: not x)
        deprecate_renamed_attribute('disable_gelu', 'disable_bias_gelu_fusion')
        deprecate_renamed_attribute('disable_layer_norm', 'disable_layer_norm_fusion')
        deprecate_renamed_attribute('disable_attention', 'disable_attention_fusion')
        deprecate_renamed_attribute('disable_skip_layer_norm', 'disable_skip_layer_norm_fusion')
        deprecate_renamed_attribute('disable_bias_skip_layer_norm', 'disable_bias_skip_layer_norm_fusion')
        deprecate_renamed_attribute('disable_bias_gelu', 'disable_bias_gelu_fusion')
        deprecate_renamed_attribute('disable_embed_layer_norm', 'disable_embed_layer_norm_fusion')

    def create_fusion_options(self, model_type: str) -> FusionOptions:

        class Box:
            pass
        args = Box()
        args.model_type = model_type
        attribute_map = {'disable_gelu_fusion': 'disable_gelu', 'disable_layer_norm_fusion': 'disable_layer_norm', 'disable_attention_fusion': 'disable_attention', 'disable_skip_layer_norm_fusion': 'disable_skip_layer_norm', 'disable_bias_skip_layer_norm_fusion': 'disable_bias_skip_layer_norm', 'disable_bias_gelu_fusion': 'disable_bias_gelu', 'disable_embed_layer_norm_fusion': 'disable_embed_layer_norm', 'disable_group_norm_fusion': 'disable_group_norm', 'disable_packed_kv': 'disable_packed_kv', 'use_raw_attention_mask': 'use_raw_attention_mask', 'enable_gemm_fast_gelu_fusion': 'enable_gemm_fast_gelu', 'use_multi_head_attention': 'use_multi_head_attention', 'disable_rotary_embeddings': 'disable_rotary_embeddings'}
        for attr_name, fusion_attr_name in attribute_map.items():
            setattr(args, fusion_attr_name, getattr(self, attr_name))
        for attr, value in self.__dict__.items():
            if hasattr(args, attr):
                continue
            setattr(args, attr, value)
        return FusionOptions.parse(args)