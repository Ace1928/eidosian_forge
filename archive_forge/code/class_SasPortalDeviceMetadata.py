from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SasPortalDeviceMetadata(_messages.Message):
    """Device data overridable by both SAS Portal and registration requests.

  Fields:
    antennaModel: If populated, the Antenna Model Pattern to use. Format is:
      `RecordCreatorId:PatternId`
    commonChannelGroup: Common Channel Group (CCG). A group of CBSDs in the
      same ICG requesting a common primary channel assignment. For more
      details, see [CBRSA-TS-2001 V3.0.0](https://ongoalliance.org/wp-
      content/uploads/2020/02/CBRSA-TS-2001-V3.0.0_Approved-for-
      publication.pdf).
    interferenceCoordinationGroup: Interference Coordination Group (ICG). A
      group of CBSDs that manage their own interference with the group. For
      more details, see [CBRSA-TS-2001 V3.0.0](https://ongoalliance.org/wp-
      content/uploads/2020/02/CBRSA-TS-2001-V3.0.0_Approved-for-
      publication.pdf).
    nrqzValidated: Output only. Set to `true` if a CPI has validated that they
      have coordinated with the National Quiet Zone office.
    nrqzValidation: Output only. National Radio Quiet Zone validation info.
  """
    antennaModel = _messages.StringField(1)
    commonChannelGroup = _messages.StringField(2)
    interferenceCoordinationGroup = _messages.StringField(3)
    nrqzValidated = _messages.BooleanField(4)
    nrqzValidation = _messages.MessageField('SasPortalNrqzValidation', 5)