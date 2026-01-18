from google.protobuf import message
import requests
from absl import logging
from tensorboard import version
from tensorboard.plugins.scalar import metadata as scalars_metadata
from tensorboard.uploader.proto import server_info_pb2
class CommunicationError(RuntimeError):
    """Raised upon failure to communicate with the server."""
    pass