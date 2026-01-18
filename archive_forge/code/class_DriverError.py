from octavia_lib.i18n import _
class DriverError(Exception):
    """Catch all exception that drivers can raise.

    This exception includes two strings: The user fault string and the
    optional operator fault string. The user fault string,
    "user_fault_string", will be provided to the API requester. The operator
    fault string, "operator_fault_string",  will be logged in the Octavia API
    log file for the operator to use when debugging.

    :param user_fault_string: String provided to the API requester.
    :type user_fault_string: string
    :param operator_fault_string: Optional string logged by the Octavia API
      for the operator to use when debugging.
    :type operator_fault_string: string
    """
    user_fault_string = _('An unknown driver error occurred.')
    operator_fault_string = _('An unknown driver error occurred.')

    def __init__(self, *args, **kwargs):
        self.user_fault_string = kwargs.pop('user_fault_string', self.user_fault_string)
        self.operator_fault_string = kwargs.pop('operator_fault_string', self.operator_fault_string)
        super().__init__(self.user_fault_string, *args, **kwargs)