from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExportMessagesRequest(_messages.Message):
    """Request to schedule an export.

  Fields:
    endTime: The end of the range in `send_time` (MSH.7, https://www.hl7.org/d
      ocumentcenter/public_temp_2E58C1F9-1C23-BA17-
      0C6126475344DA9D/wg/conf/HL7MSH.htm) to process. If not specified, the
      time when the export is scheduled is used. This value has to be after
      the `start_time` defined above. Only messages whose `send_times` lie in
      the range defined by this value and the `start_time` above are exported.
    gcsDestination: A GoogleCloudHealthcareV1alpha2Hl7v2GcsDestination
      attribute.
    startTime: The start of the range in `send_time` (MSH.7, https://www.hl7.o
      rg/documentcenter/public_temp_2E58C1F9-1C23-BA17-
      0C6126475344DA9D/wg/conf/HL7MSH.htm) to process. If not specified, the
      UNIX epoch (1970-01-01T00:00:00Z) is used. This value has to come before
      the `end_time` defined below. Only messages whose `send_times` lie in
      the range defined by this value and `end_time` are exported.
  """
    endTime = _messages.StringField(1)
    gcsDestination = _messages.MessageField('GoogleCloudHealthcareV1alpha2Hl7v2GcsDestination', 2)
    startTime = _messages.StringField(3)