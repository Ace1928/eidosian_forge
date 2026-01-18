import struct
from oslo_log import log as logging
@property
def actual_size(self):
    """Returns the total size of the file, usually smaller than
        virtual_size.
        """
    return self._total_count