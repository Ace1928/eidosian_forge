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
def RecordSetsFromZoneFile(zone_file, domain, api_version='v1'):
    """Returns record-sets for the given domain imported from the given zone file.

  Args:
    zone_file: file, The zone file with records for the given domain.
    domain: str, The domain for which record-sets should be obtained.
    api_version: [str], the api version to use for creating the records.

  Returns:
    A (name, type) keyed dict of ResourceRecordSets that were obtained from the
    zone file. Note that only records of supported types are retrieved. Also,
    the primary NS field for SOA records is discarded since that is
    provided by Cloud DNS.
  """
    zone_contents = zone.from_file(zone_file, domain, check_origin=False)
    record_sets = {}
    for name, rdset in zone_contents.iterate_rdatasets():
        record_set = _RecordSetFromZoneRecord(name, rdset, zone_contents.origin, api_version=api_version)
        if record_set:
            record_sets[record_set.name, record_set.type] = record_set
    return record_sets