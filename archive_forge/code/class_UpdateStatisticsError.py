from octavia_lib.i18n import _
class UpdateStatisticsError(Exception):
    """Exception raised when a statistics update fails.

    Each exception will include a message field that describes the
    error and references to the failed record if available.
    :param fault_string: String describing the fault.
    :type fault_string: string
    :param status_object: The object the fault occurred on.
    :type status_object: string
    :param status_object_id: The ID of the object that failed stats update.
    :type status_object_id: string
    :param status_record: The stats update record that caused the fault.
    :type status_record: string
    """
    fault_string = _('The statistics update had an unknown error.')
    stats_object = None
    stats_object_id = None
    stats_record = None

    def __init__(self, *args, **kwargs):
        self.fault_string = kwargs.pop('fault_string', self.fault_string)
        self.stats_object = kwargs.pop('stats_object', None)
        self.stats_object_id = kwargs.pop('stats_object_id', None)
        self.stats_record = kwargs.pop('stats_record', None)
        super().__init__(self.fault_string, *args, **kwargs)