from collections import OrderedDict
import contextlib
from typing import Dict, Any
from tensorboard.compat.proto.config_pb2 import RunMetadata
from tensorboard.compat.proto.graph_pb2 import GraphDef
from tensorboard.compat.proto.step_stats_pb2 import StepStats, DeviceStepStats
from tensorboard.compat.proto.versions_pb2 import VersionDef
import torch
from ._proto_graph import node_proto
Convert graph representation of GraphPy object to TensorBoard required format.