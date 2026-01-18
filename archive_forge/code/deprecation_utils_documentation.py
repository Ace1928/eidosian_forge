from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
Decorator that marks a GCloud command as deprecated.

  Args:
      remove_version: string, The GCloud sdk version where this command will be
      marked as removed.

      remove: boolean, True if the command should be removed in underlying
      base.Deprecate decorator, False if it should only print a warning

      alt_command: string, optional alternative command to use in place of
      deprecated command

  Raises:
      ValueError: If remove version is missing

  Returns:
    A modified version of the provided class.
  