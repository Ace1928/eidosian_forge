from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSecuritycenterV2AccessReview(_messages.Message):
    """Conveys information about a Kubernetes access review (such as one
  returned by a [`kubectl auth
  can-i`](https://kubernetes.io/docs/reference/access-authn-
  authz/authorization/#checking-api-access) command) that was involved in a
  finding.

  Fields:
    group: The API group of the resource. "*" means all.
    name: The name of the resource being requested. Empty means all.
    ns: Namespace of the action being requested. Currently, there is no
      distinction between no namespace and all namespaces. Both are
      represented by "" (empty).
    resource: The optional resource type requested. "*" means all.
    subresource: The optional subresource type.
    verb: A Kubernetes resource API verb, like get, list, watch, create,
      update, delete, proxy. "*" means all.
    version: The API version of the resource. "*" means all.
  """
    group = _messages.StringField(1)
    name = _messages.StringField(2)
    ns = _messages.StringField(3)
    resource = _messages.StringField(4)
    subresource = _messages.StringField(5)
    verb = _messages.StringField(6)
    version = _messages.StringField(7)