import sys
import time
import grpc
from tensorboard.data.experimental import base_experiment
from tensorboard.data.experimental import utils as experimental_utils
from tensorboard.uploader import auth
from tensorboard.uploader import util
from tensorboard.uploader import server_info as server_info_lib
from tensorboard.uploader.proto import export_service_pb2
from tensorboard.uploader.proto import export_service_pb2_grpc
from tensorboard.uploader.proto import server_info_pb2
from tensorboard.util import grpc_util
Constructor of ExperimentFromDev.

        Args:
          experiment_id: String ID of the experiment on tensorboard.dev (e.g.,
            "AdYd1TgeTlaLWXx6I8JUbA").
          api_endpoint: Optional override value for API endpoint. Used for
            development only.
        