from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class LocationPreference(_messages.Message):
    """Preferred location. This specifies where a Cloud SQL instance is
  located. Note that if the preferred location is not available, the instance
  will be located as close as possible within the region. Only one location
  may be specified.

  Fields:
    followGaeApplication: The App Engine application to follow, it must be in
      the same region as the Cloud SQL instance. WARNING: Changing this might
      restart the instance.
    kind: This is always `sql#locationPreference`.
    secondaryZone: The preferred Compute Engine zone for the
      secondary/failover (for example: us-central1-a, us-central1-b, etc.). To
      disable this field, set it to 'no_secondary_zone'.
    zone: The preferred Compute Engine zone (for example: us-central1-a, us-
      central1-b, etc.). WARNING: Changing this might restart the instance.
  """
    followGaeApplication = _messages.StringField(1)
    kind = _messages.StringField(2)
    secondaryZone = _messages.StringField(3)
    zone = _messages.StringField(4)