from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
class ManagedZone(_messages.Message):
    """A zone is a subtree of the DNS namespace under one administrative
  responsibility. A ManagedZone is a resource that represents a DNS zone
  hosted by the Cloud DNS service.

  Fields:
    creationTime: The time that this resource was created on the server. This
      is in RFC3339 text format. Output only.
    description: A mutable string of at most 1024 characters associated with
      this resource for the user's convenience. Has no effect on the managed
      zone's function.
    dnsName: The DNS name of this managed zone, for instance "example.com.".
    id: Unique identifier for the resource; defined by the server (output
      only)
    kind: Identifies what kind of resource this is. Value: the fixed string
      "dns#managedZone".
    name: User assigned name for this resource. Must be unique within the
      project. The name must be 1-32 characters long, must begin with a
      letter, end with a letter or digit, and only contain lowercase letters,
      digits or dashes.
    nameServerSet: Optionally specifies the NameServerSet for this
      ManagedZone. A NameServerSet is a set of DNS name servers that all host
      the same ManagedZones. Most users will leave this field unset.
    nameServers: Delegate your managed_zone to these virtual name servers;
      defined by the server (output only)
  """
    creationTime = _messages.StringField(1)
    description = _messages.StringField(2)
    dnsName = _messages.StringField(3)
    id = _messages.IntegerField(4, variant=_messages.Variant.UINT64)
    kind = _messages.StringField(5, default=u'dns#managedZone')
    name = _messages.StringField(6)
    nameServerSet = _messages.StringField(7)
    nameServers = _messages.StringField(8, repeated=True)