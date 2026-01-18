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
def ComputeChange(current, to_be_imported, replace_all=False, origin=None, replace_origin_ns=False, api_version='v1'):
    """Returns a change for importing the given record-sets.

  Args:
    current: dict, (name, type) keyed dict of current record-sets.
    to_be_imported: dict, (name, type) keyed dict of record-sets to be imported.
    replace_all: bool, Whether the record-sets to be imported should replace the
      current record-sets.
    origin: string, the name of the apex zone ex. "foo.com"
    replace_origin_ns: bool, Whether origin NS records should be imported.
    api_version: [str], the api version to use for creating the records.

  Raises:
    ConflictingRecordsFound: If conflicting records are found.

  Returns:
    A Change that describes the actions required to import the given
    record-sets.
  """
    messages = core_apis.GetMessagesModule('dns', api_version)
    change = messages.Change()
    change.additions = []
    change.deletions = []
    current_keys = set(current.keys())
    keys_to_be_imported = set(to_be_imported.keys())
    intersecting_keys = current_keys.intersection(keys_to_be_imported)
    if not replace_all and intersecting_keys:
        raise ConflictingRecordsFound('The following records (name type) already exist: {0}'.format([_NameAndType(current[key]) for key in sorted(intersecting_keys)]))
    for key in intersecting_keys:
        current_record = current[key]
        record_to_be_imported = to_be_imported[key]
        rdtype = _ToStandardEnumTypeSafe(key[1])
        if not _FilterOutRecord(current_record.name, rdtype, origin, replace_origin_ns):
            replacement = _GetRDataReplacement(rdtype)(current_record, record_to_be_imported, api_version=api_version)
            if replacement:
                change.deletions.append(current_record)
                change.additions.append(replacement)
    for key in keys_to_be_imported.difference(current_keys):
        change.additions.append(to_be_imported[key])
    for key in current_keys.difference(keys_to_be_imported):
        current_record = current[key]
        rdtype = _ToStandardEnumTypeSafe(key[1])
        if rdtype is rdatatype.SOA:
            change.deletions.append(current_record)
            change.additions.append(NextSOARecordSet(current_record, api_version))
        elif replace_all and (not _FilterOutRecord(current_record.name, rdtype, origin, replace_origin_ns)):
            change.deletions.append(current_record)
    if IsOnlySOAIncrement(change, api_version):
        return None
    change.additions.sort(key=_NameAndType)
    change.deletions.sort(key=_NameAndType)
    return change