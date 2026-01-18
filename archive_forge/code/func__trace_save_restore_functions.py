from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import nest
def _trace_save_restore_functions(saveable_factory, obj):
    """Traces save and restore functions."""
    if saveable_object_util.is_factory_for_restored_saveable_object(saveable_factory):
        return (saveable_factory.keywords['save_function'], saveable_factory.keywords['restore_function'])
    saveables = []

    @def_function.function(input_signature=[tensor_spec.TensorSpec([], dtypes.string)])
    def save_fn(checkpoint_key):
        maybe_saveable = saveable_factory(name=checkpoint_key)
        if isinstance(maybe_saveable, saveable_object.SaveableObject):
            maybe_saveable = [maybe_saveable]
        saveables[:] = maybe_saveable
        ret = []
        for saveable in saveables:
            for spec in saveable.specs:
                ret.append({'name': spec.name, 'tensor': spec.tensor, 'slice_spec': spec.slice_spec})
        return ret
    concrete_save = save_fn.get_concrete_function()
    saveables = saveable_object_util.validate_saveables_for_saved_model(saveables, obj)
    if not saveables:
        return (None, None)
    restored_type_specs = []
    tensor_structure = []
    for saveable in saveables:
        saveable_tensor_structure = []
        tensor_structure.append(saveable_tensor_structure)
        for spec in saveable.specs:
            restored_type_specs.append(type_spec.type_spec_from_value(spec.tensor))
            saveable_tensor_structure.append(spec.name)

    @def_function.function(input_signature=restored_type_specs)
    def restore_fn(*restored_tensors):
        structured_restored_tensors = nest.pack_sequence_as(tensor_structure, restored_tensors)
        for saveable, restored_tensors in zip(saveables, structured_restored_tensors):
            saveable.restore(restored_tensors, restored_shapes=None)
        return 1
    concrete_restore = restore_fn.get_concrete_function()
    return (concrete_save, concrete_restore)