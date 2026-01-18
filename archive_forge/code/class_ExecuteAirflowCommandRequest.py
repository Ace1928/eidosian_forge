from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExecuteAirflowCommandRequest(_messages.Message):
    """Execute Airflow Command request.

  Fields:
    command: Airflow command.
    parameters: Parameters for the Airflow command/subcommand as an array of
      arguments. It may contain positional arguments like `["my-dag-id"]`,
      key-value parameters like `["--foo=bar"]` or `["--foo","bar"]`, or other
      flags like `["-f"]`.
    subcommand: Airflow subcommand.
  """
    command = _messages.StringField(1)
    parameters = _messages.StringField(2, repeated=True)
    subcommand = _messages.StringField(3)