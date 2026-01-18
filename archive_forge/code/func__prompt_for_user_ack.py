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
def _prompt_for_user_ack(intent):
    """Prompts for user consent, exiting the program if they decline."""
    body = intent.get_ack_message_body()
    header = '\n***** TensorBoard Uploader *****\n'
    user_ack_message = '\n'.join((header, body, _MESSAGE_TOS))
    sys.stderr.write(user_ack_message)
    sys.stderr.write('\n')
    response = input('Continue? (yes/NO) ')
    if response.lower() not in ('y', 'yes'):
        sys.exit(0)
    sys.stderr.write('\n')