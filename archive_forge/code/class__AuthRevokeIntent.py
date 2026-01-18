import abc
import os
import sys
import textwrap
from absl import logging
import grpc
from tensorboard.compat import tf
from tensorboard.uploader.proto import experiment_pb2
from tensorboard.uploader.proto import export_service_pb2_grpc
from tensorboard.uploader.proto import write_service_pb2_grpc
from tensorboard.uploader import auth
from tensorboard.uploader import dry_run_stubs
from tensorboard.uploader import exporter as exporter_lib
from tensorboard.uploader import flags_parser
from tensorboard.uploader import formatters
from tensorboard.uploader import server_info as server_info_lib
from tensorboard.uploader import uploader as uploader_lib
from tensorboard.uploader.proto import server_info_pb2
from tensorboard import program
from tensorboard.plugins import base_plugin
class _AuthRevokeIntent(_Intent):
    """The user intends to revoke credentials."""

    def get_ack_message_body(self):
        """Must not be called."""
        raise AssertionError('No user ack needed to revoke credentials')

    def execute(self, server_info, channel):
        """Execute handled specially by `main`.

        Must not be called.
        """
        raise AssertionError('_AuthRevokeIntent should not be directly executed')