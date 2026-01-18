import ctypes
import enum
import os
import platform
import sys
import numpy as np
@_tf_export('lite.Interpreter')
class Interpreter:
    """Interpreter interface for running TensorFlow Lite models.

  Models obtained from `TfLiteConverter` can be run in Python with
  `Interpreter`.

  As an example, lets generate a simple Keras model and convert it to TFLite
  (`TfLiteConverter` also supports other input formats with `from_saved_model`
  and `from_concrete_function`)

  >>> x = np.array([[1.], [2.]])
  >>> y = np.array([[2.], [4.]])
  >>> model = tf.keras.models.Sequential([
  ...           tf.keras.layers.Dropout(0.2),
  ...           tf.keras.layers.Dense(units=1, input_shape=[1])
  ...         ])
  >>> model.compile(optimizer='sgd', loss='mean_squared_error')
  >>> model.fit(x, y, epochs=1)
  >>> converter = tf.lite.TFLiteConverter.from_keras_model(model)
  >>> tflite_model = converter.convert()

  `tflite_model` can be saved to a file and loaded later, or directly into the
  `Interpreter`. Since TensorFlow Lite pre-plans tensor allocations to optimize
  inference, the user needs to call `allocate_tensors()` before any inference.

  >>> interpreter = tf.lite.Interpreter(model_content=tflite_model)
  >>> interpreter.allocate_tensors()  # Needed before execution!

  Sample execution:

  >>> output = interpreter.get_output_details()[0]  # Model has single output.
  >>> input = interpreter.get_input_details()[0]  # Model has single input.
  >>> input_data = tf.constant(1., shape=[1, 1])
  >>> interpreter.set_tensor(input['index'], input_data)
  >>> interpreter.invoke()
  >>> interpreter.get_tensor(output['index']).shape
  (1, 1)

  Use `get_signature_runner()` for a more user-friendly inference API.
  """

    def __init__(self, model_path=None, model_content=None, experimental_delegates=None, num_threads=None, experimental_op_resolver_type=OpResolverType.AUTO, experimental_preserve_all_tensors=False, experimental_disable_delegate_clustering=False):
        """Constructor.

    Args:
      model_path: Path to TF-Lite Flatbuffer file.
      model_content: Content of model.
      experimental_delegates: Experimental. Subject to change. List of
        [TfLiteDelegate](https://www.tensorflow.org/lite/performance/delegates)
        objects returned by lite.load_delegate().
      num_threads: Sets the number of threads used by the interpreter and
        available to CPU kernels. If not set, the interpreter will use an
        implementation-dependent default number of threads. Currently, only a
        subset of kernels, such as conv, support multi-threading. num_threads
        should be >= -1. Setting num_threads to 0 has the effect to disable
        multithreading, which is equivalent to setting num_threads to 1. If set
        to the value -1, the number of threads used will be
        implementation-defined and platform-dependent.
      experimental_op_resolver_type: The op resolver used by the interpreter. It
        must be an instance of OpResolverType. By default, we use the built-in
        op resolver which corresponds to tflite::ops::builtin::BuiltinOpResolver
        in C++.
      experimental_preserve_all_tensors: If true, then intermediate tensors used
        during computation are preserved for inspection, and if the passed op
        resolver type is AUTO or BUILTIN, the type will be changed to
        BUILTIN_WITHOUT_DEFAULT_DELEGATES so that no Tensorflow Lite default
        delegates are applied. If false, getting intermediate tensors could
        result in undefined values or None, especially when the graph is
        successfully modified by the Tensorflow Lite default delegate.
      experimental_disable_delegate_clustering: If true, don't perform delegate
        clustering during delegate graph partitioning phase. Disabling delegate
        clustering will make the execution order of ops respect the
        explicitly-inserted control dependencies in the graph (inserted via
        `with tf.control_dependencies()`) since the TF Lite converter will drop
        control dependencies by default. Most users shouldn't turn this flag to
        True if they don't insert explicit control dependencies or the graph
        execution order is expected. For automatically inserted control
        dependencies (with `tf.Variable`, `tf.Print` etc), the user doesn't need
        to turn this flag to True since they are respected by default. Note that
        this flag is currently experimental, and it might be removed/updated if
        the TF Lite converter doesn't drop such control dependencies in the
        model. Default is False.

    Raises:
      ValueError: If the interpreter was unable to create.
    """
        if not hasattr(self, '_custom_op_registerers'):
            self._custom_op_registerers = []
        actual_resolver_type = experimental_op_resolver_type
        if experimental_preserve_all_tensors and (experimental_op_resolver_type == OpResolverType.AUTO or experimental_op_resolver_type == OpResolverType.BUILTIN):
            actual_resolver_type = OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        op_resolver_id = _get_op_resolver_id(actual_resolver_type)
        if op_resolver_id is None:
            raise ValueError('Unrecognized passed in op resolver type: {}'.format(experimental_op_resolver_type))
        if model_path and (not model_content):
            custom_op_registerers_by_name = [x for x in self._custom_op_registerers if isinstance(x, str)]
            custom_op_registerers_by_func = [x for x in self._custom_op_registerers if not isinstance(x, str)]
            self._interpreter = _interpreter_wrapper.CreateWrapperFromFile(model_path, op_resolver_id, custom_op_registerers_by_name, custom_op_registerers_by_func, experimental_preserve_all_tensors, experimental_disable_delegate_clustering)
            if not self._interpreter:
                raise ValueError('Failed to open {}'.format(model_path))
        elif model_content and (not model_path):
            custom_op_registerers_by_name = [x for x in self._custom_op_registerers if isinstance(x, str)]
            custom_op_registerers_by_func = [x for x in self._custom_op_registerers if not isinstance(x, str)]
            self._model_content = model_content
            self._interpreter = _interpreter_wrapper.CreateWrapperFromBuffer(model_content, op_resolver_id, custom_op_registerers_by_name, custom_op_registerers_by_func, experimental_preserve_all_tensors, experimental_disable_delegate_clustering)
        elif not model_content and (not model_path):
            raise ValueError('`model_path` or `model_content` must be specified.')
        else:
            raise ValueError("Can't both provide `model_path` and `model_content`")
        if num_threads is not None:
            if not isinstance(num_threads, int):
                raise ValueError('type of num_threads should be int')
            if num_threads < 1:
                raise ValueError('num_threads should >= 1')
            self._interpreter.SetNumThreads(num_threads)
        self._delegates = []
        if experimental_delegates:
            self._delegates = experimental_delegates
            for delegate in self._delegates:
                self._interpreter.ModifyGraphWithDelegate(delegate._get_native_delegate_pointer())
        self._signature_defs = self.get_signature_list()
        self._metrics = metrics.TFLiteMetrics()
        self._metrics.increase_counter_interpreter_creation()

    def __del__(self):
        self._interpreter = None
        self._delegates = None

    def allocate_tensors(self):
        self._ensure_safe()
        return self._interpreter.AllocateTensors()

    def _safe_to_run(self):
        """Returns true if there exist no numpy array buffers.

    This means it is safe to run tflite calls that may destroy internally
    allocated memory. This works, because in the wrapper.cc we have made
    the numpy base be the self._interpreter.
    """
        return sys.getrefcount(self._interpreter) == 2

    def _ensure_safe(self):
        """Makes sure no numpy arrays pointing to internal buffers are active.

    This should be called from any function that will call a function on
    _interpreter that may reallocate memory e.g. invoke(), ...

    Raises:
      RuntimeError: If there exist numpy objects pointing to internal memory
        then we throw.
    """
        if not self._safe_to_run():
            raise RuntimeError('There is at least 1 reference to internal data\n      in the interpreter in the form of a numpy array or slice. Be sure to\n      only hold the function returned from tensor() if you are using raw\n      data access.')

    def _get_op_details(self, op_index):
        """Gets a dictionary with arrays of ids for tensors involved with an op.

    Args:
      op_index: Operation/node index of node to query.

    Returns:
      a dictionary containing the index, op name, and arrays with lists of the
      indices for the inputs and outputs of the op/node.
    """
        op_index = int(op_index)
        op_name = self._interpreter.NodeName(op_index)
        op_inputs = self._interpreter.NodeInputs(op_index)
        op_outputs = self._interpreter.NodeOutputs(op_index)
        details = {'index': op_index, 'op_name': op_name, 'inputs': op_inputs, 'outputs': op_outputs}
        return details

    def _get_tensor_details(self, tensor_index, subgraph_index):
        """Gets tensor details.

    Args:
      tensor_index: Tensor index of tensor to query.
      subgraph_index: Index of the subgraph.

    Returns:
      A dictionary containing the following fields of the tensor:
        'name': The tensor name.
        'index': The tensor index in the interpreter.
        'shape': The shape of the tensor.
        'quantization': Deprecated, use 'quantization_parameters'. This field
            only works for per-tensor quantization, whereas
            'quantization_parameters' works in all cases.
        'quantization_parameters': The parameters used to quantize the tensor:
          'scales': List of scales (one if per-tensor quantization)
          'zero_points': List of zero_points (one if per-tensor quantization)
          'quantized_dimension': Specifies the dimension of per-axis
              quantization, in the case of multiple scales/zero_points.

    Raises:
      ValueError: If tensor_index is invalid.
    """
        tensor_index = int(tensor_index)
        subgraph_index = int(subgraph_index)
        tensor_name = self._interpreter.TensorName(tensor_index, subgraph_index)
        tensor_size = self._interpreter.TensorSize(tensor_index, subgraph_index)
        tensor_size_signature = self._interpreter.TensorSizeSignature(tensor_index, subgraph_index)
        tensor_type = self._interpreter.TensorType(tensor_index, subgraph_index)
        tensor_quantization = self._interpreter.TensorQuantization(tensor_index, subgraph_index)
        tensor_quantization_params = self._interpreter.TensorQuantizationParameters(tensor_index, subgraph_index)
        tensor_sparsity_params = self._interpreter.TensorSparsityParameters(tensor_index, subgraph_index)
        if not tensor_type:
            raise ValueError('Could not get tensor details')
        details = {'name': tensor_name, 'index': tensor_index, 'shape': tensor_size, 'shape_signature': tensor_size_signature, 'dtype': tensor_type, 'quantization': tensor_quantization, 'quantization_parameters': {'scales': tensor_quantization_params[0], 'zero_points': tensor_quantization_params[1], 'quantized_dimension': tensor_quantization_params[2]}, 'sparsity_parameters': tensor_sparsity_params}
        return details

    def _get_ops_details(self):
        """Gets op details for every node.

    Returns:
      A list of dictionaries containing arrays with lists of tensor ids for
      tensors involved in the op.
    """
        return [self._get_op_details(idx) for idx in range(self._interpreter.NumNodes())]

    def get_tensor_details(self):
        """Gets tensor details for every tensor with valid tensor details.

    Tensors where required information about the tensor is not found are not
    added to the list. This includes temporary tensors without a name.

    Returns:
      A list of dictionaries containing tensor information.
    """
        tensor_details = []
        for idx in range(self._interpreter.NumTensors(0)):
            try:
                tensor_details.append(self._get_tensor_details(idx, subgraph_index=0))
            except ValueError:
                pass
        return tensor_details

    def get_input_details(self):
        """Gets model input tensor details.

    Returns:
      A list in which each item is a dictionary with details about
      an input tensor. Each dictionary contains the following fields
      that describe the tensor:

      + `name`: The tensor name.
      + `index`: The tensor index in the interpreter.
      + `shape`: The shape of the tensor.
      + `shape_signature`: Same as `shape` for models with known/fixed shapes.
        If any dimension sizes are unknown, they are indicated with `-1`.
      + `dtype`: The numpy data type (such as `np.int32` or `np.uint8`).
      + `quantization`: Deprecated, use `quantization_parameters`. This field
        only works for per-tensor quantization, whereas
        `quantization_parameters` works in all cases.
      + `quantization_parameters`: A dictionary of parameters used to quantize
        the tensor:
        ~ `scales`: List of scales (one if per-tensor quantization).
        ~ `zero_points`: List of zero_points (one if per-tensor quantization).
        ~ `quantized_dimension`: Specifies the dimension of per-axis
        quantization, in the case of multiple scales/zero_points.
      + `sparsity_parameters`: A dictionary of parameters used to encode a
        sparse tensor. This is empty if the tensor is dense.
    """
        return [self._get_tensor_details(i, subgraph_index=0) for i in self._interpreter.InputIndices()]

    def set_tensor(self, tensor_index, value):
        """Sets the value of the input tensor.

    Note this copies data in `value`.

    If you want to avoid copying, you can use the `tensor()` function to get a
    numpy buffer pointing to the input buffer in the tflite interpreter.

    Args:
      tensor_index: Tensor index of tensor to set. This value can be gotten from
        the 'index' field in get_input_details.
      value: Value of tensor to set.

    Raises:
      ValueError: If the interpreter could not set the tensor.
    """
        self._interpreter.SetTensor(tensor_index, value)

    def resize_tensor_input(self, input_index, tensor_size, strict=False):
        """Resizes an input tensor.

    Args:
      input_index: Tensor index of input to set. This value can be gotten from
        the 'index' field in get_input_details.
      tensor_size: The tensor_shape to resize the input to.
      strict: Only unknown dimensions can be resized when `strict` is True.
        Unknown dimensions are indicated as `-1` in the `shape_signature`
        attribute of a given tensor. (default False)

    Raises:
      ValueError: If the interpreter could not resize the input tensor.

    Usage:
    ```
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.resize_tensor_input(0, [num_test_images, 224, 224, 3])
    interpreter.allocate_tensors()
    interpreter.set_tensor(0, test_images)
    interpreter.invoke()
    ```
    """
        self._ensure_safe()
        tensor_size = np.array(tensor_size, dtype=np.int32)
        self._interpreter.ResizeInputTensor(input_index, tensor_size, strict)

    def get_output_details(self):
        """Gets model output tensor details.

    Returns:
      A list in which each item is a dictionary with details about
      an output tensor. The dictionary contains the same fields as
      described for `get_input_details()`.
    """
        return [self._get_tensor_details(i, subgraph_index=0) for i in self._interpreter.OutputIndices()]

    def get_signature_list(self):
        """Gets list of SignatureDefs in the model.

    Example,
    ```
    signatures = interpreter.get_signature_list()
    print(signatures)

    # {
    #   'add': {'inputs': ['x', 'y'], 'outputs': ['output_0']}
    # }

    Then using the names in the signature list you can get a callable from
    get_signature_runner().
    ```

    Returns:
      A list of SignatureDef details in a dictionary structure.
      It is keyed on the SignatureDef method name, and the value holds
      dictionary of inputs and outputs.
    """
        full_signature_defs = self._interpreter.GetSignatureDefs()
        for _, signature_def in full_signature_defs.items():
            signature_def['inputs'] = list(signature_def['inputs'].keys())
            signature_def['outputs'] = list(signature_def['outputs'].keys())
        return full_signature_defs

    def _get_full_signature_list(self):
        """Gets list of SignatureDefs in the model.

    Example,
    ```
    signatures = interpreter._get_full_signature_list()
    print(signatures)

    # {
    #   'add': {'inputs': {'x': 1, 'y': 0}, 'outputs': {'output_0': 4}}
    # }

    Then using the names in the signature list you can get a callable from
    get_signature_runner().
    ```

    Returns:
      A list of SignatureDef details in a dictionary structure.
      It is keyed on the SignatureDef method name, and the value holds
      dictionary of inputs and outputs.
    """
        return self._interpreter.GetSignatureDefs()

    def get_signature_runner(self, signature_key=None):
        """Gets callable for inference of specific SignatureDef.

    Example usage,
    ```
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    fn = interpreter.get_signature_runner('div_with_remainder')
    output = fn(x=np.array([3]), y=np.array([2]))
    print(output)
    # {
    #   'quotient': array([1.], dtype=float32)
    #   'remainder': array([1.], dtype=float32)
    # }
    ```

    None can be passed for signature_key if the model has a single Signature
    only.

    All names used are this specific SignatureDef names.


    Args:
      signature_key: Signature key for the SignatureDef, it can be None if and
        only if the model has a single SignatureDef. Default value is None.

    Returns:
      This returns a callable that can run inference for SignatureDef defined
      by argument 'signature_key'.
      The callable will take key arguments corresponding to the arguments of the
      SignatureDef, that should have numpy values.
      The callable will returns dictionary that maps from output names to numpy
      values of the computed results.

    Raises:
      ValueError: If passed signature_key is invalid.
    """
        if signature_key is None:
            if len(self._signature_defs) != 1:
                raise ValueError('SignatureDef signature_key is None and model has {0} Signatures. None is only allowed when the model has 1 SignatureDef'.format(len(self._signature_defs)))
            else:
                signature_key = next(iter(self._signature_defs))
        return SignatureRunner(interpreter=self, signature_key=signature_key)

    def get_tensor(self, tensor_index, subgraph_index=0):
        """Gets the value of the output tensor (get a copy).

    If you wish to avoid the copy, use `tensor()`. This function cannot be used
    to read intermediate results.

    Args:
      tensor_index: Tensor index of tensor to get. This value can be gotten from
        the 'index' field in get_output_details.
      subgraph_index: Index of the subgraph to fetch the tensor. Default value
        is 0, which means to fetch from the primary subgraph.

    Returns:
      a numpy array.
    """
        return self._interpreter.GetTensor(tensor_index, subgraph_index)

    def tensor(self, tensor_index):
        """Returns function that gives a numpy view of the current tensor buffer.

    This allows reading and writing to this tensors w/o copies. This more
    closely mirrors the C++ Interpreter class interface's tensor() member, hence
    the name. Be careful to not hold these output references through calls
    to `allocate_tensors()` and `invoke()`. This function cannot be used to read
    intermediate results.

    Usage:

    ```
    interpreter.allocate_tensors()
    input = interpreter.tensor(interpreter.get_input_details()[0]["index"])
    output = interpreter.tensor(interpreter.get_output_details()[0]["index"])
    for i in range(10):
      input().fill(3.)
      interpreter.invoke()
      print("inference %s" % output())
    ```

    Notice how this function avoids making a numpy array directly. This is
    because it is important to not hold actual numpy views to the data longer
    than necessary. If you do, then the interpreter can no longer be invoked,
    because it is possible the interpreter would resize and invalidate the
    referenced tensors. The NumPy API doesn't allow any mutability of the
    the underlying buffers.

    WRONG:

    ```
    input = interpreter.tensor(interpreter.get_input_details()[0]["index"])()
    output = interpreter.tensor(interpreter.get_output_details()[0]["index"])()
    interpreter.allocate_tensors()  # This will throw RuntimeError
    for i in range(10):
      input.fill(3.)
      interpreter.invoke()  # this will throw RuntimeError since input,output
    ```

    Args:
      tensor_index: Tensor index of tensor to get. This value can be gotten from
        the 'index' field in get_output_details.

    Returns:
      A function that can return a new numpy array pointing to the internal
      TFLite tensor state at any point. It is safe to hold the function forever,
      but it is not safe to hold the numpy array forever.
    """
        return lambda: self._interpreter.tensor(self._interpreter, tensor_index)

    def invoke(self):
        """Invoke the interpreter.

    Be sure to set the input sizes, allocate tensors and fill values before
    calling this. Also, note that this function releases the GIL so heavy
    computation can be done in the background while the Python interpreter
    continues. No other function on this object should be called while the
    invoke() call has not finished.

    Raises:
      ValueError: When the underlying interpreter fails raise ValueError.
    """
        self._ensure_safe()
        self._interpreter.Invoke()

    def reset_all_variables(self):
        return self._interpreter.ResetVariableTensors()

    def _native_handle(self):
        """Returns a pointer to the underlying tflite::Interpreter instance.

    This allows extending tflite.Interpreter's functionality in a custom C++
    function. Consider how that may work in a custom pybind wrapper:

      m.def("SomeNewFeature", ([](py::object handle) {
        auto* interpreter =
          reinterpret_cast<tflite::Interpreter*>(handle.cast<intptr_t>());
        ...
      }))

    and corresponding Python call:

      SomeNewFeature(interpreter.native_handle())

    Note: This approach is fragile. Users must guarantee the C++ extension build
    is consistent with the tflite.Interpreter's underlying C++ build.
    """
        return self._interpreter.interpreter()