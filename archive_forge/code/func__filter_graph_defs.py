import contextlib
import functools
import time
import grpc
from google.protobuf import message
from tensorboard.compat.proto import graph_pb2
from tensorboard.compat.proto import summary_pb2
from tensorboard.compat.proto import types_pb2
from tensorboard.uploader.proto import write_service_pb2
from tensorboard.uploader import logdir_loader
from tensorboard.uploader import upload_tracker
from tensorboard.uploader import util
from tensorboard.backend import process_graph
from tensorboard.backend.event_processing import directory_loader
from tensorboard.backend.event_processing import event_file_loader
from tensorboard.backend.event_processing import io_wrapper
from tensorboard.plugins.graph import metadata as graphs_metadata
from tensorboard.util import grpc_util
from tensorboard.util import tb_logging
from tensorboard.util import tensor_util
def _filter_graph_defs(event):
    for v in event.summary.value:
        if v.metadata.plugin_data.plugin_name != graphs_metadata.PLUGIN_NAME:
            continue
        if v.tag == graphs_metadata.RUN_GRAPH_NAME:
            data = list(v.tensor.string_val)
            filtered_data = [_filtered_graph_bytes(x) for x in data]
            filtered_data = [x for x in filtered_data if x is not None]
            if filtered_data != data:
                new_tensor = tensor_util.make_tensor_proto(filtered_data, dtype=types_pb2.DT_STRING)
                v.tensor.CopyFrom(new_tensor)