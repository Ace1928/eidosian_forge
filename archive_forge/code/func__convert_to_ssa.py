import copy
import logging
import os
import re
from tensorboard.compat.proto.graph_pb2 import GraphDef
from tensorboard.compat.proto.node_def_pb2 import NodeDef
from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace
from typing import Set, Dict, Tuple, List
def _convert_to_ssa(shapes, blob_name_tracker, ops):
    """
    Convert an operator graph to SSA (i.e. out-of-place).

    i.e. blobs will be renamed so that each blob is produced only once.

    Args:
        shapes: Dictionary mapping blob names to their shapes/dimensions.
        blob_name_tracker: Dictionary of all unique blob names (with respect to
            some context).
        ops: List of Caffe2 operators

    Returns:
        None. Modifies blob_name_tracker and ops in-place.
    """
    ir = core.IR(ops)
    seen: Set[str] = set()
    versioned: Dict[Tuple[str, int], int] = {}
    new_shapes = {}
    new_blob_name_tracker = {}

    def ssa_name(name: str, versions: Dict[str, int]) -> int:
        assert name in versions
        version = versions[name]
        if (name, version) in versioned:
            return versioned[name, version]
        new_name = _make_unique_name(seen, name, min_version=version)
        versioned[name, version] = new_name
        if name in shapes:
            new_shapes[new_name] = shapes[name]
        if blob_name_tracker and name in blob_name_tracker:
            new_blob_name_tracker[new_name] = blob_name_tracker[name]
        return new_name
    for op, ssa in zip(ops, ir.ssa):
        assert op is ssa.op
        inputs = list(op.input)
        outputs = list(op.output)
        del op.input[:]
        del op.output[:]
        op.input.extend((ssa_name(name, ssa.in_versions) for name in inputs))
        op.output.extend((ssa_name(name, ssa.out_versions) for name in outputs))
    shapes.clear()
    shapes.update(new_shapes)
    if blob_name_tracker:
        blob_name_tracker.clear()
        blob_name_tracker.update(new_blob_name_tracker)