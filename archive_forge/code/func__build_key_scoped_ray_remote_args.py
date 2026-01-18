import atexit
import threading
from collections import defaultdict
from collections import OrderedDict
from dataclasses import dataclass
from multiprocessing.pool import ThreadPool
from typing import Optional
import ray
import dask
from dask.core import istask, ishashable, _execute_task
from dask.system import CPU_COUNT
from dask.threaded import pack_exception, _thread_get_id
from ray.util.dask.callbacks import local_ray_callbacks, unpack_ray_callbacks
from ray.util.dask.common import unpack_object_refs
from ray.util.dask.scheduler_utils import get_async, apply_sync
def _build_key_scoped_ray_remote_args(dsk, annotations, ray_remote_args):
    if not isinstance(dsk, dask.highlevelgraph.HighLevelGraph):
        dsk = dask.highlevelgraph.HighLevelGraph.from_collections(id(dsk), dsk, dependencies=())
    scoped_annotations = {}
    layers = [(name, dsk.layers[name]) for name in dsk._toposort_layers()]
    for id_, layer in layers:
        layer_annotations = layer.annotations
        if layer_annotations is None:
            layer_annotations = annotations
        elif 'resources' in layer_annotations:
            raise ValueError(TOP_LEVEL_RESOURCES_ERR_MSG)
        for key in layer.get_output_keys():
            layer_annotations_for_key = annotations.copy()
            layer_annotations_for_key.update(layer_annotations)
            layer_annotations_for_key.update(scoped_annotations.get(key, {}))
            scoped_annotations[key] = layer_annotations_for_key
    scoped_ray_remote_args = {}
    for key, annotations in scoped_annotations.items():
        layer_ray_remote_args = ray_remote_args.copy()
        layer_ray_remote_args.update(annotations.get('ray_remote_args', {}))
        scoped_ray_remote_args[key] = layer_ray_remote_args
    return scoped_ray_remote_args