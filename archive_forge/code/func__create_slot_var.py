from tensorflow.python.compiler.xla.experimental import xla_sharding
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import ref_variable
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
def _create_slot_var(primary, val, scope, validate_shape, shape, dtype, *, copy_xla_sharding=False):
    """Helper function for creating a slot variable."""
    current_partitioner = variable_scope.get_variable_scope().partitioner
    variable_scope.get_variable_scope().set_partitioner(None)
    shape = shape if callable(val) else None
    if resource_variable_ops.is_resource_variable(primary):
        use_resource = True
    elif isinstance(primary, ref_variable.RefVariable):
        use_resource = False
    else:
        use_resource = None
    slot = variable_scope.get_variable(scope, initializer=val, trainable=False, use_resource=use_resource, shape=shape, dtype=dtype, validate_shape=validate_shape)
    variable_scope.get_variable_scope().set_partitioner(current_partitioner)
    if isinstance(primary, variables.Variable) and primary._save_slice_info:
        real_slot_name = slot.name[len(primary.op.name + '/'):-2]
        slice_info = primary._save_slice_info
        n = slot.shape.ndims
        if n is None or n > 0:
            slot._set_save_slice_info(variables.Variable.SaveSliceInfo(slice_info.full_name + '/' + real_slot_name, slice_info.full_shape[:n], slice_info.var_offset[:n], slice_info.var_shape[:n]))

    def _has_same_rank(primary_shape, slot_shape):
        return primary_shape.rank is not None and slot_shape.rank is not None and (primary_shape.rank == slot_shape.rank)
    if copy_xla_sharding and _has_same_rank(primary.shape, slot.shape):
        slot = xla_sharding.copy_sharding(primary, slot, use_sharding_op=False)
    return slot