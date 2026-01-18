from octavia_lib.i18n import _
class UnsupportedOptionError(Exception):
    """Exception raised when a driver does not support an option.

    Provider drivers will validate that they can complete the request -- that
    all options are supported by the driver. If the request fails validation,
    drivers will raise an UnsupportedOptionError exception. For example, if a
    driver does not support a flavor passed as an option to load balancer
    create(), the driver will raise an UnsupportedOptionError and include a
    message parameter providing an explanation of the failure.

    :param user_fault_string: String provided to the API requester.
    :type user_fault_string: string
    :param operator_fault_string: Optional string logged by the Octavia API
      for the operator to use when debugging.
    :type operator_fault_string: string
    """
    user_fault_string = _('A specified option is not supported by this provider.')
    operator_fault_string = _('A specified option is not supported by this provider.')

    def __init__(self, *args, **kwargs):
        self.user_fault_string = kwargs.pop('user_fault_string', self.user_fault_string)
        self.operator_fault_string = kwargs.pop('operator_fault_string', self.operator_fault_string)
        super().__init__(self.user_fault_string, *args, **kwargs)