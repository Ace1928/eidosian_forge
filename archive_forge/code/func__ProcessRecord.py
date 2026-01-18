from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core.resource import resource_printer_base
from googlecloudsdk.core.util import encoding
import six
def _ProcessRecord(self, record):
    """Applies process_record_orig to dict, list and default repr records.

    Args:
      record: A JSON-serializable object.

    Returns:
      The processed record.
    """
    if isinstance(record, (dict, list)) or _HasDefaultRepr(record):
        record = self._process_record_orig(record)
    if isinstance(record, dict):
        return ['{0}: {1}'.format(k, v) for k, v in sorted(six.iteritems(record)) if v is not None]
    if isinstance(record, list):
        return [i for i in record if i is not None]
    return [encoding.Decode(record or '')]