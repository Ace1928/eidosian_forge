import enum
import functools
import pprint
import shutil
import sys
import tempfile
import time
import warnings
from absl import logging
from google.protobuf import text_format as _text_format
from google.protobuf.message import DecodeError
from tensorflow.core.framework import graph_pb2 as _graph_pb2
from tensorflow.lite.experimental.microfrontend.python.ops import audio_microfrontend_op  # pylint: disable=unused-import
from tensorflow.lite.python import conversion_metadata_schema_py_generated as conversion_metdata_fb
from tensorflow.lite.python import lite_constants as constants
from tensorflow.lite.python.convert import convert_graphdef as _convert_graphdef
from tensorflow.lite.python.convert import convert_graphdef_with_arrays as _convert_graphdef_with_arrays
from tensorflow.lite.python.convert import convert_jax_hlo as _convert_jax_hlo
from tensorflow.lite.python.convert import convert_saved_model as _convert_saved_model
from tensorflow.lite.python.convert import ConverterError  # pylint: disable=unused-import
from tensorflow.lite.python.convert import deduplicate_readonly_buffers as _deduplicate_readonly_buffers
from tensorflow.lite.python.convert import mlir_quantize as _mlir_quantize
from tensorflow.lite.python.convert import mlir_sparsify as _mlir_sparsify
from tensorflow.lite.python.convert import OpsSet
from tensorflow.lite.python.convert import toco_convert  # pylint: disable=unused-import
from tensorflow.lite.python.convert_phase import Component
from tensorflow.lite.python.convert_phase import convert_phase
from tensorflow.lite.python.convert_phase import SubComponent
from tensorflow.lite.python.convert_saved_model import freeze_saved_model as _freeze_saved_model
from tensorflow.lite.python.interpreter import Interpreter  # pylint: disable=unused-import
from tensorflow.lite.python.interpreter import load_delegate  # pylint: disable=unused-import
from tensorflow.lite.python.interpreter import OpResolverType  # pylint: disable=unused-import
from tensorflow.lite.python.metrics import metrics
from tensorflow.lite.python.op_hint import convert_op_hints_to_stubs  # pylint: disable=unused-import
from tensorflow.lite.python.op_hint import is_ophint_converted as _is_ophint_converted
from tensorflow.lite.python.op_hint import OpHint  # pylint: disable=unused-import
from tensorflow.lite.python.optimize import calibrator as _calibrator
from tensorflow.lite.python.util import _xla_computation
from tensorflow.lite.python.util import build_debug_info_func as _build_debug_info_func
from tensorflow.lite.python.util import convert_debug_info_func as _convert_debug_info_func
from tensorflow.lite.python.util import freeze_graph as _freeze_graph
from tensorflow.lite.python.util import get_debug_info as _get_debug_info
from tensorflow.lite.python.util import get_grappler_config as _get_grappler_config
from tensorflow.lite.python.util import get_sparsity_modes as _get_sparsity_modes
from tensorflow.lite.python.util import get_tensor_name as _get_tensor_name
from tensorflow.lite.python.util import get_tensors_from_tensor_names as _get_tensors_from_tensor_names
from tensorflow.lite.python.util import get_tf_type_name as _get_tf_type_name
from tensorflow.lite.python.util import is_frozen_graph as _is_frozen_graph
from tensorflow.lite.python.util import model_input_signature as _model_input_signature
from tensorflow.lite.python.util import modify_model_io_type as _modify_model_io_type
from tensorflow.lite.python.util import populate_conversion_metadata as _populate_conversion_metadata
from tensorflow.lite.python.util import run_graph_optimizations as _run_graph_optimizations
from tensorflow.lite.python.util import set_tensor_shapes as _set_tensor_shapes
from tensorflow.lite.python.util import trace_model_call as _trace_model_call
from tensorflow.lite.tools import flatbuffer_utils
from tensorflow.lite.tools.optimize.debugging.python.debugger import QuantizationDebugger  # pylint: disable=unused-import
from tensorflow.lite.tools.optimize.debugging.python.debugger import QuantizationDebugOptions  # pylint: disable=unused-import
from tensorflow.python import saved_model as _saved_model
from tensorflow.python.client import session as _session
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function as _def_function
from tensorflow.python.eager import function as _function
from tensorflow.python.framework import byte_swap_tensor as bst
from tensorflow.python.framework import convert_to_constants as _convert_to_constants
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import versions
from tensorflow.python.framework.errors_impl import NotFoundError as _NotFoundError
from tensorflow.python.framework.importer import import_graph_def as _import_graph_def
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import loader_impl as _loader_impl
from tensorflow.python.saved_model import save_options as _save_options
from tensorflow.python.saved_model import signature_constants as _signature_constants
from tensorflow.python.saved_model import tag_constants as _tag_constants
from tensorflow.python.saved_model.load import load as _load
from tensorflow.python.saved_model.loader_impl import parse_saved_model_with_debug_info as _parse_saved_model_with_debug_info
from tensorflow.python.util import deprecation as _deprecation
from tensorflow.python.util import keras_deps
from tensorflow.python.util.tf_export import tf_export as _tf_export
@_tf_export('lite.TFLiteConverter', v1=[])
class TFLiteConverterV2(TFLiteFrozenGraphConverterV2):
    """Converts a TensorFlow model into TensorFlow Lite model.

  Attributes:
    optimizations: Experimental flag, subject to change. Set of optimizations to
      apply. e.g {tf.lite.Optimize.DEFAULT}. (default None, must be None or a
      set of values of type `tf.lite.Optimize`)
    representative_dataset: A generator function used for integer quantization
      where each generated sample has the same order, type and shape as the
      inputs to the model. Usually, this is a small subset of a few hundred
      samples randomly chosen, in no particular order, from the training or
      evaluation dataset. This is an optional attribute, but required for full
      integer quantization, i.e, if `tf.int8` is the only supported type in
      `target_spec.supported_types`. Refer to `tf.lite.RepresentativeDataset`.
      (default None)
    target_spec: Experimental flag, subject to change. Specifications of target
      device, including supported ops set, supported types and a set of user's
      defined TensorFlow operators required in the TensorFlow Lite runtime.
      Refer to `tf.lite.TargetSpec`.
    inference_input_type: Data type of the input layer. Note that integer types
      (tf.int8 and tf.uint8) are currently only supported for post training
      integer quantization and quantization aware training. (default tf.float32,
      must be in {tf.float32, tf.int8, tf.uint8})
    inference_output_type: Data type of the output layer. Note that integer
      types (tf.int8 and tf.uint8) are currently only supported for post
      training integer quantization and quantization aware training. (default
      tf.float32, must be in {tf.float32, tf.int8, tf.uint8})
    allow_custom_ops: Boolean indicating whether to allow custom operations.
      When False, any unknown operation is an error. When True, custom ops are
      created for any op that is unknown. The developer needs to provide these
      to the TensorFlow Lite runtime with a custom resolver. (default False)
    exclude_conversion_metadata: Whether not to embed the conversion metadata
      into the converted model. (default False)
    experimental_new_converter: Experimental flag, subject to change. Enables
      MLIR-based conversion. (default True)
    experimental_new_quantizer: Experimental flag, subject to change. Enables
      MLIR-based quantization conversion instead of Flatbuffer-based conversion.
      (default True)
    experimental_enable_resource_variables: Experimental flag, subject to
      change. Enables [resource
      variables](https://tensorflow.org/guide/migrate/tf1_vs_tf2#resourcevariables_instead_of_referencevariables)
      to be converted by this converter. This is only allowed if the
      from_saved_model interface is used. (default True)

  Example usage:

  ```python
  # Converting a SavedModel to a TensorFlow Lite model.
  converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
  tflite_model = converter.convert()

  # Converting a tf.Keras model to a TensorFlow Lite model.
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  tflite_model = converter.convert()

  # Converting ConcreteFunctions to a TensorFlow Lite model.
  converter = tf.lite.TFLiteConverter.from_concrete_functions([func], model)
  tflite_model = converter.convert()

  # Converting a Jax model to a TensorFlow Lite model.
  converter = tf.lite.TFLiteConverter.experimental_from_jax(
      [func], [[ ('input1', input1), ('input2', input2)]])
  tflite_model = converter.convert()
  ```
  """

    def __init__(self, funcs, trackable_obj=None):
        """Constructor for TFLiteConverter.

    Args:
      funcs: List of TensorFlow ConcreteFunctions. The list should not contain
        duplicate elements.
      trackable_obj: tf.AutoTrackable object associated with `funcs`. A
        reference to this object needs to be maintained so that Variables do not
        get garbage collected since functions have a weak reference to
        Variables. This is only required when the tf.AutoTrackable object is not
        maintained by the user (e.g. `from_saved_model`).
    """
        super(TFLiteConverterV2, self).__init__(funcs, trackable_obj)

    @classmethod
    def from_concrete_functions(cls, funcs, trackable_obj=None):
        """Creates a TFLiteConverter object from ConcreteFunctions.

    Args:
      funcs: List of TensorFlow ConcreteFunctions. The list should not contain
        duplicate elements. Currently converter can only convert a single
        ConcreteFunction. Converting multiple functions is under development.
      trackable_obj:   An `AutoTrackable` object (typically `tf.module`)
        associated with `funcs`. A reference to this object needs to be
        maintained so that Variables do not get garbage collected since
        functions have a weak reference to Variables.

    Returns:
      TFLiteConverter object.

    Raises:
      Invalid input type.
    """
        TFLiteConverterBase._set_original_model_type(conversion_metdata_fb.ModelType.TF_CONCRETE_FUNCTIONS)
        if trackable_obj is None:
            logging.warning('Please consider providing the trackable_obj argument in the from_concrete_functions. Providing without the trackable_obj argument is deprecated and it will use the deprecated conversion path.')
        for func in funcs:
            if not isinstance(func, _function.ConcreteFunction):
                message = 'This function takes in a list of ConcreteFunction.'
                if isinstance(func, _def_function.Function):
                    message += ' To get the ConcreteFunction from a Function, call get_concrete_function.'
                raise ValueError(message)
        return cls(funcs, trackable_obj)

    @classmethod
    def from_saved_model(cls, saved_model_dir, signature_keys=None, tags=None):
        """Creates a TFLiteConverter object from a SavedModel directory.

    Args:
      saved_model_dir: SavedModel directory to convert.
      signature_keys: List of keys identifying SignatureDef containing inputs
        and outputs. Elements should not be duplicated. By default the
        `signatures` attribute of the MetaGraphdef is used. (default
        saved_model.signatures)
      tags: Set of tags identifying the MetaGraphDef within the SavedModel to
        analyze. All tags in the tag set must be present. (default
        {tf.saved_model.SERVING} or {'serve'})

    Returns:
      TFLiteConverter object.

    Raises:
      Invalid signature keys.
    """
        TFLiteConverterBase._set_original_model_type(conversion_metdata_fb.ModelType.TF_SAVED_MODEL)
        if not context.executing_eagerly():
            signature_key = None
            if signature_keys:
                if len(signature_keys) != 1:
                    raise ValueError('Only support a single signature key.')
                else:
                    signature_key = signature_keys[0]
            logging.warning('Invoking the TF1 implementation of TFLiteConverter because eager is disabled. Consider enabling eager.')
            return TFLiteConverter.from_saved_model(saved_model_dir, signature_key=signature_key, tag_set=tags)
        if tags is None:
            tags = set([_tag_constants.SERVING])
        with context.eager_mode():
            saved_model = _load(saved_model_dir, tags)
        if not signature_keys:
            signature_keys = saved_model.signatures
        if not signature_keys:
            raise ValueError('Only support at least one signature key.')
        if len(signature_keys) > 1 and hasattr(saved_model, 'serve') and (not hasattr(saved_model, '_default_save_signature')):
            saved_model.serving_default = saved_model.serve
            delattr(saved_model, 'serve')
            signature_keys = ['serving_default']
        funcs = []
        for key in signature_keys:
            if key not in saved_model.signatures:
                raise ValueError("Invalid signature key '{}' found. Valid keys are '{}'.".format(key, ','.join(saved_model.signatures)))
            funcs.append(saved_model.signatures[key])
        saved_model_converter = TFLiteSavedModelConverterV2(saved_model_dir, tags, signature_keys, saved_model)
        if saved_model_converter.saved_model_dir:
            return saved_model_converter
        return cls(funcs, saved_model)

    @classmethod
    def from_keras_model(cls, model):
        """Creates a TFLiteConverter object from a Keras model.

    Args:
      model: tf.Keras.Model

    Returns:
      TFLiteConverter object.
    """
        TFLiteConverterBase._set_original_model_type(conversion_metdata_fb.ModelType.KERAS_MODEL)
        return TFLiteKerasModelConverterV2(model)

    @classmethod
    @_deprecation.deprecated(None, 'Use `jax2tf.convert` and (`lite.TFLiteConverter.from_saved_model` or `lite.TFLiteConverter.from_concrete_functions`) instead.')
    def experimental_from_jax(cls, serving_funcs, inputs):
        """Creates a TFLiteConverter object from a Jax model with its inputs.

    Args:
      serving_funcs: A array of Jax functions with all the weights applied
        already.
      inputs: A array of Jax input placeholders tuples list, e.g.,
        jnp.zeros(INPUT_SHAPE). Each tuple list should correspond with the
        serving function.

    Returns:
      TFLiteConverter object.
    """
        TFLiteConverterBase._set_original_model_type(conversion_metdata_fb.ModelType.JAX)
        return TFLiteJaxConverterV2(serving_funcs, inputs)

    def convert(self):
        """Converts a TensorFlow GraphDef based on instance variables.

    Returns:
      The converted data in serialized format.

    Raises:
      ValueError:
        No concrete functions is specified.
        Multiple concrete functions are specified.
        Input shape is not specified.
        Invalid quantization parameters.
    """
        return super(TFLiteConverterV2, self).convert()