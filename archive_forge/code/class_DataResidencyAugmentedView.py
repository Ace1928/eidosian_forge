from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataResidencyAugmentedView(_messages.Message):
    """Next tag: 7

  Fields:
    crGopoGuris: Cloud resource to Google owned production object mapping in
      the form of GURIs. The GURIs should be available in DG KB storage/cns
      tables. This is the preferred way of providing cloud resource mappings.
      For further details please read go/cloud-resource-monitoring_sig
    crGopoPrefixes: Cloud resource to Google owned production object mapping
      in the form of prefixes. These should be available in DG KB storage/cns
      tables. The entity type, which is the part of the string before the
      first colon in the GURI, must be completely specified in prefix. For
      details about GURI please read go/guri. For further details about the
      field please read go/cloud-resource-monitoring_sig.
    serviceData: Service-specific data. Only required for pre-determined
      services. Generally used to bind a Cloud Resource to some a TI container
      that uniquely specifies a customer. See milestone 2 of DRZ KR8 SIG for
      more information.
    tpIds: The list of project_id's of the tenant projects in the 'google.com'
      org which serve the Cloud Resource. See go/drz-mst-sig for more details.
  """
    crGopoGuris = _messages.StringField(1, repeated=True)
    crGopoPrefixes = _messages.StringField(2, repeated=True)
    serviceData = _messages.MessageField('ServiceData', 3)
    tpIds = _messages.StringField(4, repeated=True)