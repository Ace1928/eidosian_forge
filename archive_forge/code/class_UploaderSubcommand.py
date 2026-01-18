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
class UploaderSubcommand(program.TensorBoardSubcommand):
    """Integration point with `tensorboard` CLI."""

    def __init__(self, experiment_url_callback=None):
        """Constructor of UploaderSubcommand.

        Args:
          experiment_url_callback: A function accepting a single string argument
            containing the full TB.dev URL of the uploaded experiment.
        """
        self._experiment_url_callback = experiment_url_callback

    def name(self):
        return 'dev'

    def define_flags(self, parser):
        flags_parser.define_flags(parser)

    def run(self, flags):
        return _run(flags, self._experiment_url_callback)

    def help(self):
        return 'upload data to TensorBoard.dev'