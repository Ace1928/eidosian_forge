import collections
import glob
import os.path
import threading
import time
import numpy as np
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.core.protobuf import trackable_object_graph_pb2
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import errors
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.trackable import base as trackable
from tensorflow.python.training import py_checkpoint_reader
from tensorflow.python.training import training_util
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export
class BaseSaverBuilder:
    """Base class for Savers.

  Can be extended to create different Ops.
  """
    SaveSpec = saveable_object.SaveSpec
    SaveableObject = saveable_object.SaveableObject
    VariableSaveable = saveable_object_util.ReferenceVariableSaveable
    ResourceVariableSaveable = saveable_object_util.ResourceVariableSaveable

    def __init__(self, write_version=saver_pb2.SaverDef.V2):
        self._write_version = write_version

    def save_op(self, filename_tensor, saveables):
        """Create an Op to save 'saveables'.

    This is intended to be overridden by subclasses that want to generate
    different Ops.

    Args:
      filename_tensor: String Tensor.
      saveables: A list of BaseSaverBuilder.SaveableObject objects.

    Returns:
      An Operation that save the variables.

    Raises:
      RuntimeError: (implementation detail) if "self._write_version" is an
        unexpected value.
    """
        tensor_names = []
        tensors = []
        tensor_slices = []
        for saveable in saveables:
            for spec in saveable.specs:
                tensor_names.append(spec.name)
                tensors.append(spec.tensor)
                tensor_slices.append(spec.slice_spec)
        if self._write_version == saver_pb2.SaverDef.V1:
            return io_ops._save(filename=filename_tensor, tensor_names=tensor_names, tensors=tensors, tensor_slices=tensor_slices)
        elif self._write_version == saver_pb2.SaverDef.V2:
            return io_ops.save_v2(filename_tensor, tensor_names, tensor_slices, tensors)
        else:
            raise RuntimeError('Unexpected write_version: ' + self._write_version)

    def bulk_restore(self, filename_tensor, saveables, preferred_shard, restore_sequentially):
        """Restore all tensors contained in saveables.

    By default, this issues separate calls to `restore_op` for each saveable.
    Subclasses may override to load multiple saveables in a single call.

    Args:
      filename_tensor: String Tensor.
      saveables: List of BaseSaverBuilder.SaveableObject objects.
      preferred_shard: Int.  Shard to open first when loading a sharded file.
      restore_sequentially: Unused.  Bool.  If true, each restore is sequential.

    Returns:
      A list of Tensors resulting from reading 'saveable' from
        'filename'.

    """
        del restore_sequentially
        all_tensors = []
        for saveable in saveables:
            if saveable.device:
                device = saveable_object_util.set_cpu0(saveable.device)
            else:
                device = None
            with ops.device(device):
                all_tensors.extend(self.restore_op(filename_tensor, saveable, preferred_shard))
        return all_tensors

    def restore_op(self, filename_tensor, saveable, preferred_shard):
        """Create ops to restore 'saveable'.

    This is intended to be overridden by subclasses that want to generate
    different Ops.

    Args:
      filename_tensor: String Tensor.
      saveable: A BaseSaverBuilder.SaveableObject object.
      preferred_shard: Int.  Shard to open first when loading a sharded file.

    Returns:
      A list of Tensors resulting from reading 'saveable' from
        'filename'.
    """
        tensors = []
        for spec in saveable.specs:
            tensors.append(io_ops.restore_v2(filename_tensor, [spec.name], [spec.slice_spec], [spec.dtype])[0])
        return tensors

    def sharded_filename(self, filename_tensor, shard, num_shards):
        """Append sharding information to a filename.

    Args:
      filename_tensor: A string tensor.
      shard: Integer.  The shard for the filename.
      num_shards: An int Tensor for the number of shards.

    Returns:
      A string tensor.
    """
        return gen_io_ops.sharded_filename(filename_tensor, shard, num_shards)

    def _AddSaveOps(self, filename_tensor, saveables):
        """Add ops to save variables that are on the same shard.

    Args:
      filename_tensor: String Tensor.
      saveables: A list of SaveableObject objects.

    Returns:
      A tensor with the filename used to save.
    """
        save = self.save_op(filename_tensor, saveables)
        return control_flow_ops.with_dependencies([save], filename_tensor)

    def _AddShardedSaveOpsForV2(self, checkpoint_prefix, per_device):
        """Add ops to save the params per shard, for the V2 format.

    Note that the sharded save procedure for the V2 format is different from
    V1: there is a special "merge" step that merges the small metadata produced
    from each device.

    Args:
      checkpoint_prefix: scalar String Tensor.  Interpreted *NOT AS A FILENAME*,
        but as a prefix of a V2 checkpoint;
      per_device: A list of (device, BaseSaverBuilder.VarToSave) pairs, as
        returned by _GroupByDevices().

    Returns:
      An op to save the variables, which, when evaluated, returns the prefix
        "<user-fed prefix>" only and does not include the sharded spec suffix.
    """
        with ops.device('CPU'):
            _SHARDED_SUFFIX = array_ops.where(string_ops.regex_full_match(checkpoint_prefix, '^s3://.*'), constant_op.constant('.part'), constant_op.constant(os.path.normpath('_temp/part')))
            tmp_checkpoint_prefix = string_ops.string_join([checkpoint_prefix, _SHARDED_SUFFIX])
        num_shards = len(per_device)
        sharded_saves = []
        sharded_prefixes = []
        num_shards_tensor = constant_op.constant(num_shards, name='num_shards')
        last_device = None
        for shard, (device, saveables) in enumerate(per_device):
            last_device = device
            with ops.device(saveable_object_util.set_cpu0(device)):
                sharded_filename = self.sharded_filename(tmp_checkpoint_prefix, shard, num_shards_tensor)
                sharded_prefixes.append(sharded_filename)
                sharded_saves.append(self._AddSaveOps(sharded_filename, saveables))
        with ops.control_dependencies([x.op for x in sharded_saves]):
            with ops.device(saveable_object_util.set_cpu0(last_device)):
                merge_step = gen_io_ops.merge_v2_checkpoints(sharded_prefixes, checkpoint_prefix, delete_old_dirs=True)
                with ops.control_dependencies([merge_step]):
                    return array_ops.identity(checkpoint_prefix)

    def _AddShardedSaveOps(self, filename_tensor, per_device):
        """Add ops to save the params per shard.

    Args:
      filename_tensor: a scalar String Tensor.
      per_device: A list of (device, BaseSaverBuilder.SaveableObject) pairs, as
        returned by _GroupByDevices().

    Returns:
      An op to save the variables.
    """
        if self._write_version == saver_pb2.SaverDef.V2:
            return self._AddShardedSaveOpsForV2(filename_tensor, per_device)
        num_shards = len(per_device)
        sharded_saves = []
        num_shards_tensor = constant_op.constant(num_shards, name='num_shards')
        for shard, (device, saveables) in enumerate(per_device):
            with ops.device(device):
                sharded_filename = self.sharded_filename(filename_tensor, shard, num_shards_tensor)
                sharded_saves.append(self._AddSaveOps(sharded_filename, saveables))
        with ops.control_dependencies([x.op for x in sharded_saves]):
            return gen_io_ops.sharded_filespec(filename_tensor, num_shards_tensor)

    def _AddRestoreOps(self, filename_tensor, saveables, restore_sequentially, reshape, preferred_shard=-1, name='restore_all'):
        """Add operations to restore saveables.

    Args:
      filename_tensor: Tensor for the path of the file to load.
      saveables: A list of SaveableObject objects.
      restore_sequentially: True if we want to restore variables sequentially
        within a shard.
      reshape: True if we want to reshape loaded tensors to the shape of the
        corresponding variable.
      preferred_shard: Shard to open first when loading a sharded file.
      name: Name for the returned op.

    Returns:
      An Operation that restores the variables.
    """
        all_tensors = self.bulk_restore(filename_tensor, saveables, preferred_shard, restore_sequentially)
        assign_ops = []
        idx = 0
        for saveable in saveables:
            shapes = None
            if reshape:
                shapes = []
                for spec in saveable.specs:
                    v = spec.tensor
                    shape = v.get_shape()
                    if not shape.is_fully_defined():
                        shape = array_ops.shape(v)
                    shapes.append(shape)
            saveable_tensors = all_tensors[idx:idx + len(saveable.specs)]
            idx += len(saveable.specs)
            assign_ops.append(saveable.restore(saveable_tensors, shapes))
        return control_flow_ops.group(*assign_ops, name=name)

    def _AddShardedRestoreOps(self, filename_tensor, per_device, restore_sequentially, reshape):
        """Add Ops to restore variables from multiple devices.

    Args:
      filename_tensor: Tensor for the path of the file to load.
      per_device: A list of (device, SaveableObject) pairs, as returned by
        _GroupByDevices().
      restore_sequentially: True if we want to restore variables sequentially
        within a shard.
      reshape: True if we want to reshape loaded tensors to the shape of the
        corresponding variable.

    Returns:
      An Operation that restores the variables.
    """
        sharded_restores = []
        for shard, (device, saveables) in enumerate(per_device):
            with ops.device(device):
                sharded_restores.append(self._AddRestoreOps(filename_tensor, saveables, restore_sequentially, reshape, preferred_shard=shard, name='restore_shard'))
        return control_flow_ops.group(*sharded_restores, name='restore_all')

    def _GroupByDevices(self, saveables):
        """Group Variable tensor slices per device.

    TODO(touts): Make sure that all the devices found are on different
    job/replica/task/cpu|gpu.  It would be bad if 2 were on the same device.
    It can happen if the devices are unspecified.

    Args:
      saveables: A list of BaseSaverBuilder.SaveableObject objects.

    Returns:
      A list of tuples: (device_name, BaseSaverBuilder.SaveableObject) tuples.
      The list is sorted by ascending device_name.

    Raises:
      ValueError: If the tensors of a saveable are on different devices.
    """
        per_device = collections.defaultdict(lambda: [])
        for saveable in saveables:
            canonical_device = set((pydev.canonical_name(spec.device) for spec in saveable.specs))
            if len(canonical_device) != 1:
                raise ValueError('All tensors of a saveable object must be on the same device: %s' % saveable.name)
            per_device[canonical_device.pop()].append(saveable)
        return sorted(per_device.items(), key=lambda t: t[0])

    def build(self, names_to_saveables, reshape=False, sharded=False, max_to_keep=5, keep_checkpoint_every_n_hours=10000.0, name=None, restore_sequentially=False, filename='model'):
        """Builds save/restore graph nodes or runs save/restore in eager mode.

    Args:
      names_to_saveables: A dictionary mapping name to a Variable or
        SaveableObject. Each name will be associated with the corresponding
        variable in the checkpoint.
      reshape: If True, allow restoring parameters from a checkpoint that where
        the parameters have a different shape.  This is only needed when you try
        to restore from a Dist-Belief checkpoint, and only some times.
      sharded: If True, shard the checkpoints, one per device that has Variable
        nodes.
      max_to_keep: Maximum number of checkpoints to keep.  As new checkpoints
        are created, old ones are deleted.  If None or 0, no checkpoints are
        deleted from the filesystem but only the last one is kept in the
        `checkpoint` file.  Presently the number is only roughly enforced.  For
        example in case of restarts more than max_to_keep checkpoints may be
        kept.
      keep_checkpoint_every_n_hours: How often checkpoints should be kept.
        Defaults to 10,000 hours.
      name: String.  Optional name to use as a prefix when adding operations.
      restore_sequentially: A Bool, which if true, causes restore of different
        variables to happen sequentially within each device.
      filename: If known at graph construction time, filename used for variable
        loading/saving. If None, then the default name "model" will be used.

    Returns:
      A SaverDef proto.

    Raises:
      TypeError: If 'names_to_saveables' is not a dictionary mapping string
        keys to variable Tensors.
      ValueError: If any of the keys or values in 'names_to_saveables' is not
        unique.
    """
        return self._build_internal(names_to_saveables=names_to_saveables, reshape=reshape, sharded=sharded, max_to_keep=max_to_keep, keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours, name=name, restore_sequentially=restore_sequentially, filename=filename)

    def _build_internal(self, names_to_saveables, reshape=False, sharded=False, max_to_keep=5, keep_checkpoint_every_n_hours=10000.0, name=None, restore_sequentially=False, filename='model', build_save=True, build_restore=True):
        """build() with option to only perform save and restore."""
        if not context.executing_eagerly() and (not build_save or not build_restore):
            raise ValueError('save and restore operations need to be built together  when eager execution is not enabled.')
        if not isinstance(names_to_saveables, dict):
            names_to_saveables = saveable_object_util.op_list_to_dict(names_to_saveables)
        saveables = saveable_object_util.validate_and_slice_inputs(names_to_saveables)
        if max_to_keep is None:
            max_to_keep = 0
        with ops.name_scope(name, 'save', [saveable.op for saveable in saveables]) as name:
            filename_tensor = array_ops.placeholder_with_default(filename or 'model', shape=(), name='filename')
            filename_tensor = array_ops.placeholder_with_default(filename_tensor, shape=(), name='Const')
            if sharded:
                per_device = self._GroupByDevices(saveables)
                if build_save:
                    save_tensor = self._AddShardedSaveOps(filename_tensor, per_device)
                if build_restore:
                    restore_op = self._AddShardedRestoreOps(filename_tensor, per_device, restore_sequentially, reshape)
            else:
                if build_save:
                    save_tensor = self._AddSaveOps(filename_tensor, saveables)
                if build_restore:
                    restore_op = self._AddRestoreOps(filename_tensor, saveables, restore_sequentially, reshape)
        if context.executing_eagerly():
            save_tensor_name = save_tensor.numpy() if build_save else ''
            return saver_pb2.SaverDef(filename_tensor_name=filename_tensor.numpy(), save_tensor_name=save_tensor_name, restore_op_name='', max_to_keep=max_to_keep, sharded=sharded, keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours, version=self._write_version)
        else:
            graph = ops.get_default_graph()
            check_collection_list = graph.get_all_collection_keys()
            for collection_type in check_collection_list:
                for element in graph.get_collection(collection_type):
                    if isinstance(element, variables.PartitionedVariable):
                        try:
                            graph.get_operation_by_name(element.name)
                        except KeyError:
                            element.as_tensor()
            return saver_pb2.SaverDef(filename_tensor_name=filename_tensor.name, save_tensor_name=save_tensor.name, restore_op_name=restore_op.name, max_to_keep=max_to_keep, sharded=sharded, keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours, version=self._write_version)