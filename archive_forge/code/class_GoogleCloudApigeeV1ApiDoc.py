from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ApiDoc(_messages.Message):
    """`ApiDoc` represents an API catalog item. Catalog items are used in two
  ways in a portal: - Users can browse and interact with a visual
  representation of the API documentation - The `api_product_name` field
  provides a link to a backing [API product]
  (/apigee/docs/reference/apis/apigee/rest/v1/organizations.apiproducts).
  Through this link, portal users can create and manage developer apps linked
  to one or more API products.

  Fields:
    anonAllowed: Optional. Boolean flag that manages user access to the
      catalog item. When true, the catalog item has public visibility and can
      be viewed anonymously; otherwise, only registered users may view it.
      Note: when the parent portal is enrolled in the [audience management
      feature](https://cloud.google.com/apigee/docs/api-
      platform/publish/portal/portal-audience#enrolling_in_the_beta_release_of
      _the_audience_management_feature), and this flag is set to false,
      visibility is set to an indeterminate state and must be explicitly
      specified in the management UI (see [Manage the visibility of an API in
      your portal](https://cloud.google.com/apigee/docs/api-
      platform/publish/portal/publish-apis#visibility)). Additionally, when
      enrolled in the audience management feature, updates to this flag will
      be ignored as visibility permissions must be updated in the management
      UI.
    apiProductName: Required. Immutable. The `name` field of the associated
      [API product](/apigee/docs/reference/apis/apigee/rest/v1/organizations.a
      piproducts). A portal may have only one catalog item associated with a
      given API product.
    categoryIds: Optional. The IDs of the API categories to which this catalog
      item belongs.
    description: Optional. Description of the catalog item. Max length is
      10,000 characters.
    edgeAPIProductName: Optional. Immutable. DEPRECATED: use the
      `apiProductName` field instead
    graphqlEndpointUrl: Optional. DEPRECATED: manage documentation through the
      `getDocumentation` and `updateDocumentation` methods
    graphqlSchema: Optional. DEPRECATED: manage documentation through the
      `getDocumentation` and `updateDocumentation` methods
    graphqlSchemaDisplayName: Optional. DEPRECATED: manage documentation
      through the `getDocumentation` and `updateDocumentation` methods
    id: Output only. The ID of the catalog item.
    imageUrl: Optional. Location of the image used for the catalog item in the
      catalog. For portal files, this can have the format `/files/{filename}`.
      Max length is 2,083 characters.
    modified: Output only. Time the catalog item was last modified in
      milliseconds since epoch.
    published: Optional. Denotes whether the catalog item is published to the
      portal or is in a draft state. When the parent portal is enrolled in the
      [audience management feature](https://cloud.google.com/apigee/docs/api-
      platform/publish/portal/portal-audience#enrolling_in_the_beta_release_of
      _the_audience_management_feature), the visibility can be set to public
      on creation by setting the anonAllowed flag to true or further managed
      in the management UI (see [Manage the visibility of an API in your
      portal](https://cloud.google.com/apigee/docs/api-
      platform/publish/portal/publish-apis#visibility)) before it can be
      visible to any users. If not enrolled in the audience management
      feature, the visibility is managed by the `anonAllowed` flag.
    requireCallbackUrl: Optional. Whether a callback URL is required when this
      catalog item's API product is enabled in a developer app. When true, a
      portal user will be required to input a URL when managing the app (this
      is typically used for the app's OAuth flow).
    siteId: Output only. The ID of the parent portal.
    specId: Optional. DEPRECATED: DO NOT USE
    title: Required. The user-facing name of the catalog item. `title` must be
      a non-empty string with a max length of 255 characters.
    visibility: Optional. DEPRECATED: use the `published` field instead
  """
    anonAllowed = _messages.BooleanField(1)
    apiProductName = _messages.StringField(2)
    categoryIds = _messages.StringField(3, repeated=True)
    description = _messages.StringField(4)
    edgeAPIProductName = _messages.StringField(5)
    graphqlEndpointUrl = _messages.StringField(6)
    graphqlSchema = _messages.StringField(7)
    graphqlSchemaDisplayName = _messages.StringField(8)
    id = _messages.IntegerField(9)
    imageUrl = _messages.StringField(10)
    modified = _messages.IntegerField(11)
    published = _messages.BooleanField(12)
    requireCallbackUrl = _messages.BooleanField(13)
    siteId = _messages.StringField(14)
    specId = _messages.StringField(15)
    title = _messages.StringField(16)
    visibility = _messages.BooleanField(17)