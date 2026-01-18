from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import encoding as api_encoding
from dns import rdatatype
from dns import zone
from googlecloudsdk.api_lib.dns import record_types
from googlecloudsdk.api_lib.dns import svcb_stub
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import encoding
import six
def NextSOARecordSet(soa_record_set, api_version='v1'):
    """Returns a new SOA record set with an incremented serial number.

  Args:
    soa_record_set: ResourceRecordSet, Current SOA record-set.
    api_version: [str], the api version to use for creating the records.

  Returns:
    A a new SOA record-set with an incremented serial number.
  """
    next_soa_record_set = _RecordSetCopy(soa_record_set, api_version=api_version)
    rdata_parts = soa_record_set.rrdatas[0].split()
    rdata_parts[2] = str((int(rdata_parts[2]) + 1) % (1 << 32))
    next_soa_record_set.rrdatas[0] = ' '.join(rdata_parts)
    return next_soa_record_set