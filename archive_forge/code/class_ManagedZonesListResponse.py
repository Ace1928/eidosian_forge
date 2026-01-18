from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
class ManagedZonesListResponse(_messages.Message):
    """A ManagedZonesListResponse object.

  Fields:
    kind: Type of resource.
    managedZones: The managed zone resources.
    nextPageToken: The presence of this field indicates that there exist more
      results following your last page of results in pagination order. To
      fetch them, make another list request using this value as your page
      token.  In this way you can retrieve the complete contents of even very
      large collections one page at a time. However, if the contents of the
      collection change between the first and last paginated list request, the
      set of all elements returned will be an inconsistent view of the
      collection. There is no way to retrieve a consistent snapshot of a
      collection larger than the maximum page size.
  """
    kind = _messages.StringField(1, default=u'dns#managedZonesListResponse')
    managedZones = _messages.MessageField('ManagedZone', 2, repeated=True)
    nextPageToken = _messages.StringField(3)