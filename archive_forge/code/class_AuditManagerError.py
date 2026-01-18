from googlecloudsdk.api_lib.util import exceptions as gcloud_exception
from googlecloudsdk.core import exceptions as gcloud_core_exceptions
class AuditManagerError(gcloud_core_exceptions.Error):
    """Custom error class for Audit Manager related exceptions.

  Attributes:
    http_exception: core http exception thrown by gcloud
    suggested_command_purpose: what the suggested command achieves
    suggested_command: suggested command to help fix the exception
  """

    def __init__(self, error, suggested_command_purpose=None, suggested_command=None):
        self.http_exception = gcloud_exception.HttpException(error, ERROR_FORMAT)
        self.suggested_command_purpose = suggested_command_purpose
        self.suggested_command = suggested_command

    def __str__(self):
        message = f'{self.http_exception}'
        if self.suggested_command_purpose is not None:
            message += f'\n\nRun the following command to {self.suggested_command_purpose}:\n\n{self.suggested_command}'
        return message

    @property
    def error_info(self):
        return self.http_exception.payload.type_details['ErrorInfo']

    def has_error_info(self, reason):
        return reason in [e['reason'] for e in self.error_info]