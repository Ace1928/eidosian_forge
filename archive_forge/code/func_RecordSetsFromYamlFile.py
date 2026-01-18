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
def RecordSetsFromYamlFile(yaml_file, include_extended_records=False, api_version='v1'):
    """Returns record-sets read from the given yaml file.

  Args:
    yaml_file: file, A yaml file with records.
    include_extended_records: [bool], If extended record should be included
      (otherwise they are silently skipped).
    api_version: [str], the api version to use for creating the records.

  Returns:
    A (name, type) keyed dict of ResourceRecordSets that were obtained from the
    yaml file. Note that only records of supported types are retrieved. Also,
    the primary NS field for SOA records is discarded since that is
    provided by Cloud DNS.
  """
    record_sets = {}
    messages = core_apis.GetMessagesModule('dns', api_version)
    yaml_record_sets = yaml.load_all(yaml_file)
    for yaml_record_set in yaml_record_sets:
        rdata_type = _ToStandardEnumTypeSafe(yaml_record_set['type'])
        if rdata_type not in record_types.SUPPORTED_TYPES and (not include_extended_records or yaml_record_set['type'] not in record_types.CLOUD_DNS_EXTENDED_TYPES):
            continue
        record_set = messages.ResourceRecordSet()
        record_set.kind = record_set.kind
        record_set.name = yaml_record_set['name']
        record_set.ttl = yaml_record_set['ttl']
        record_set.type = yaml_record_set['type']
        if 'rrdatas' in yaml_record_set:
            record_set.rrdatas = yaml_record_set['rrdatas']
        elif 'routingPolicy' in yaml_record_set:
            record_set.routingPolicy = api_encoding.PyValueToMessage(messages.RRSetRoutingPolicy, yaml_record_set['routingPolicy'])
        if rdata_type is rdatatype.SOA:
            record_set.rrdatas[0] = re.sub('\\S+', '{0}', record_set.rrdatas[0], count=1)
        record_sets[record_set.name, record_set.type] = record_set
    return record_sets