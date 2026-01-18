import errno
import logging
import os
import subprocess
import tempfile
import time
import grpc
import pkg_resources
from tensorboard.data import grpc_provider
from tensorboard.data import ingester
from tensorboard.data.proto import data_provider_pb2
from tensorboard.util import tb_logging
def _make_stub(addr, channel_creds_type):
    creds, options = channel_creds_type.channel_config()
    options.append(('grpc.max_receive_message_length', 1024 * 1024 * 256))
    channel = grpc.secure_channel(addr, creds, options=options)
    return grpc_provider.make_stub(channel)