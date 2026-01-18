from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRunV2VersionToPath(_messages.Message):
    """VersionToPath maps a specific version of a secret to a relative file to
  mount to, relative to VolumeMount's mount_path.

  Fields:
    mode: Integer octal mode bits to use on this file, must be a value between
      01 and 0777 (octal). If 0 or not set, the Volume's default mode will be
      used. Notes * Internally, a umask of 0222 will be applied to any non-
      zero value. * This is an integer representation of the mode bits. So,
      the octal integer value should look exactly as the chmod numeric
      notation with a leading zero. Some examples: for chmod 777 (a=rwx), set
      to 0777 (octal) or 511 (base-10). For chmod 640 (u=rw,g=r), set to 0640
      (octal) or 416 (base-10). For chmod 755 (u=rwx,g=rx,o=rx), set to 0755
      (octal) or 493 (base-10). * This might be in conflict with other options
      that affect the file mode, like fsGroup, and the result can be other
      mode bits set.
    path: Required. The relative path of the secret in the container.
    version: The Cloud Secret Manager secret version. Can be 'latest' for the
      latest value, or an integer or a secret alias for a specific version.
  """
    mode = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    path = _messages.StringField(2)
    version = _messages.StringField(3)