from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute.os_config import flags
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
class ListTableRow:
    """View model for table rows of OS policy assignment reports list."""

    def __init__(self, instance, assignment_id, location, update_time, summary_str):
        self.instance = instance
        self.assignment_id = assignment_id
        self.location = location
        self.update_time = update_time
        self.summary_str = summary_str

    def __eq__(self, other):
        return self.instance == other.instance and self.assignment_id == other.assignment_id and (self.location == other.location) and (self.update_time == other.update_time) and (self.summary_str == other.summary_str)

    def __repr__(self):
        return 'ListTableRow(instance=%s, assignment_id=%s, location=%s, update_time=%s, summary_str=%s)' % (self.instance, self.assignment_id, self.location, self.update_time, self.summary_str)