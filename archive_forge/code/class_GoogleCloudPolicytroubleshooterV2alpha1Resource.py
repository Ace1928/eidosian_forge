from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicytroubleshooterV2alpha1Resource(_messages.Message):
    """This message defines core attributes for a resource. A resource is an
  addressable (named) entity provided by the destination service. For example,
  a file stored on a network storage service.

  Fields:
    name: The stable identifier (name) of a resource on the `service`. A
      resource can be logically identified as
      "//{resource.service}/{resource.name}". The differences between a
      resource name and a URI are:  *   Resource name is a logical identifier,
      independent of network     protocol and API version. For example,
      `//pubsub.googleapis.com/projects/123/topics/news-feed`. *   URI often
      includes protocol and version information, so it can     be used
      directly by applications. For example,
      `https://pubsub.googleapis.com/v1/projects/123/topics/news-feed`.  See
      https://cloud.google.com/apis/design/resource_names for details.
    service: The name of the service that this resource belongs to, such as
      `pubsub.googleapis.com`. The service may be different from the DNS
      hostname that actually serves the request.
    type: The type of the resource. The syntax is platform-specific because
      different platforms define their resources differently.  For Google
      APIs, the type format must be "{service}/{kind}".
  """
    name = _messages.StringField(1)
    service = _messages.StringField(2)
    type = _messages.StringField(3)