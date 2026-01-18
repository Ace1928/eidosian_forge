from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ApiCategory(_messages.Message):
    """`ApiCategory` represents an API category. [Catalog items](/apigee/docs/r
  eference/apis/apigee/rest/v1/organizations.sites.apidocs) can be tagged with
  API categories; users viewing the API catalog in the portal will have the
  option to browse the catalog by category.

  Fields:
    id: ID of the category (a UUID).
    name: Name of the category.
    siteId: Name of the portal.
    updateTime: Time the category was last modified in milliseconds since
      epoch.
  """
    id = _messages.StringField(1)
    name = _messages.StringField(2)
    siteId = _messages.StringField(3)
    updateTime = _messages.IntegerField(4)