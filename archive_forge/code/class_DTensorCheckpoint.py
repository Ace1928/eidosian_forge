from typing import Dict, List, Optional
import weakref
from tensorflow.core.protobuf import trackable_object_graph_pb2
from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import d_variable
from tensorflow.dtensor.python import gen_dtensor_ops
from tensorflow.dtensor.python import layout
from tensorflow.dtensor.python import save_restore
from tensorflow.python.checkpoint import checkpoint as util
from tensorflow.python.checkpoint import checkpoint_options
from tensorflow.python.checkpoint import graph_view as graph_view_lib
from tensorflow.python.checkpoint import restore as restore_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.trackable import base
from tensorflow.python.trackable import data_structures
from tensorflow.python.training import py_checkpoint_reader
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
@deprecation.deprecated(date=None, instructions='Please use tf.train.Checkpoint instead of DTensorCheckpoint. DTensor is integrated with tf.train.Checkpoint and it can be used out of the box to save and restore dtensors.')
@tf_export('experimental.dtensor.DTensorCheckpoint', v1=[])
class DTensorCheckpoint(util.Checkpoint):
    """Manages saving/restoring trackable values to disk, for DTensor."""

    def __init__(self, mesh: layout.Mesh, root=None, **kwargs):
        super(DTensorCheckpoint, self).__init__(root=root, **kwargs)
        self._mesh = mesh
        saver_root = self
        attached_dependencies = None
        self._save_counter = None
        self._save_assign_op = None
        if root:
            util._assert_trackable(root, 'root')
            saver_root = root
            attached_dependencies = []
            kwargs['root'] = root
            root._maybe_initialize_trackable()
            self._save_counter = data_structures.NoDependency(root._lookup_dependency('save_counter'))
            self._root = data_structures.NoDependency(root)
        for k, v in sorted(kwargs.items(), key=lambda item: item[0]):
            setattr(self, k, v)
            converted_v = getattr(self, k)
            util._assert_trackable(converted_v, k)
            if root:
                attached_dependencies = attached_dependencies or []
                child = root._lookup_dependency(k)
                if child is None:
                    attached_dependencies.append(base.TrackableReference(k, converted_v))
                elif child != converted_v:
                    raise ValueError('Cannot create a Checkpoint with keyword argument {name} if root.{name} already exists.'.format(name=k))
        self._saver = DTrackableSaver(mesh, graph_view_lib.ObjectGraphView(weakref.ref(saver_root), attached_dependencies=attached_dependencies))