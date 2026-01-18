from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import shutil
from gslib import storage_url
from gslib.utils import execution_util
from gslib.utils import temporary_file_util
from boto import config
def _stet_transform(subcommand, blob_id, in_file_path, out_file_path, logger):
    """Runs a STET transform on a file.

  Encrypts for uploads. Decrypts for downloads. Automatically populates
  flags for the STET binary.

  Args:
    subcommand (StetSubcommandName): Subcommand to call on STET binary.
    blob_id (str): Cloud URL that binary uses for validation.
    in_file_path (str): File to be transformed source.
    out_file_path (str): Where to write result of transform.
    logger (logging.Logger): For logging STET binary output.

  Raises:
    KeyError: STET binary or config could not be found.
  """
    binary_path = config.get('GSUtil', 'stet_binary_path', _get_stet_binary_from_path())
    if not binary_path:
        raise KeyError('Could not find STET binary in boto config or PATH.')
    command_args = [os.path.expanduser(binary_path), subcommand]
    config_path = config.get('GSUtil', 'stet_config_path', None)
    if config_path:
        command_args.append('--config-file=' + os.path.expanduser(config_path))
    command_args.extend(['--blob-id=' + blob_id, in_file_path, out_file_path])
    _, stderr = execution_util.ExecuteExternalCommand(command_args)
    logger.debug(stderr)