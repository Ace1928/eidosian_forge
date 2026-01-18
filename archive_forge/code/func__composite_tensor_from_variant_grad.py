from tensorflow.core.protobuf import composite_tensor_variant_pb2
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import gen_composite_tensor_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util import nest
@ops.RegisterGradient('CompositeTensorVariantToComponents')
def _composite_tensor_from_variant_grad(op, *grad):
    assert len(grad) == len(op.outputs)
    components = [op.outputs[i] if grad[i] is None else grad[i] for i in range(len(grad))]
    return gen_composite_tensor_ops.CompositeTensorVariantFromComponents(components=components, metadata=op.get_attr('metadata'))