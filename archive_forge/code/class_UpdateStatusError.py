from octavia_lib.i18n import _
class UpdateStatusError(Exception):
    """Exception raised when a status update fails.

    Each exception will include a message field that describes the
    error and references to the failed record if available.
    :param fault_string: String describing the fault.
    :type fault_string: string
    :param status_object: The object the fault occurred on.
    :type status_object: string
    :param status_object_id: The ID of the object that failed status update.
    :type status_object_id: string
    :param status_record: The status update record that caused the fault.
    :type status_record: string
    """
    fault_string = _('The status update had an unknown error.')
    status_object = None
    status_object_id = None
    status_record = None

    def __init__(self, *args, **kwargs):
        self.fault_string = kwargs.pop('fault_string', self.fault_string)
        self.status_object = kwargs.pop('status_object', None)
        self.status_object_id = kwargs.pop('status_object_id', None)
        self.status_record = kwargs.pop('status_record', None)
        super().__init__(self.fault_string, *args, **kwargs)