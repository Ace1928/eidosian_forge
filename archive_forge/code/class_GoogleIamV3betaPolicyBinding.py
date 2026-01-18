from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleIamV3betaPolicyBinding(_messages.Message):
    """IAM policy binding

  Enums:
    PolicyKindValueValuesEnum: Immutable. The kind of the policy to attach in
      this binding: + When the policy is empty, this field must be set. + When
      the policy is set, this field + can be left empty and will be set to the
      policy kind, or + must set to the input policy kind

  Messages:
    AnnotationsValue: Optional. User defined annotations. See
      https://google.aip.dev/148#annotations for more details such as format
      and size limitations

  Fields:
    annotations: Optional. User defined annotations. See
      https://google.aip.dev/148#annotations for more details such as format
      and size limitations
    condition: Optional. Condition can either be a principal condition or a
      resource condition. It depends on the type of target, the policy it is
      attached to, and/or the expression itself. When set, the `expression`
      field in the `Expr` must include from 1 to 10 subexpressions, joined by
      the "||"(Logical OR), "&&"(Logical AND) or "!"(Logical NOT) operators.
      Allowed operations for principal.type: - `principal.type == ` -
      `principal.type != ` - `principal.type in []` Allowed operations for
      principal.subject: - `principal.subject == ` - `principal.subject != ` -
      `principal.subject in []` - `principal.subject.startsWith()` -
      `principal.subject.endsWith()` Supported principal types are Workspace,
      Workforce Pool, Workload Pool and Service Account. Allowed string must
      be one of: - iam.googleapis.com/WorkspaceIdentity -
      iam.googleapis.com/WorkforcePoolIdentity -
      iam.googleapis.com/WorkloadPoolIdentity -
      iam.googleapis.com/ServiceAccount When the bound policy is a Principal
      Access Boundary policy, each subexpression must be of the form
      `principal.type == ` or `principal.subject == ''`. An example expression
      is: "principal.type == 'iam.googleapis.com/ServiceAccount'" or
      "principal.subject == 'bob@acme.com'".
    createTime: Output only. The time when the policy binding was created.
    displayName: Optional. The description of the policy binding. Must be less
      than or equal to 63 characters.
    etag: Optional. The etag for the policy binding. If this is provided on
      update, it must match the server's etag.
    name: Identifier. The resource name of the policy binding. The binding
      parent is the closest CRM resource (i.e., Project, Folder or
      Organization) to the binding target. Format: `projects/{project_id}/loca
      tions/{location}/policyBindings/{policy_binding_id}` `projects/{project_
      number}/locations/{location}/policyBindings/{policy_binding_id}` `folder
      s/{folder_id}/locations/{location}/policyBindings/{policy_binding_id}` `
      organizations/{organization_id}/locations/{location}/policyBindings/{pol
      icy_binding_id}`
    policy: Required. Immutable. The resource name of the policy to be bound.
      The binding parent and policy must belong to the same Organization (or
      Project).
    policyKind: Immutable. The kind of the policy to attach in this binding: +
      When the policy is empty, this field must be set. + When the policy is
      set, this field + can be left empty and will be set to the policy kind,
      or + must set to the input policy kind
    policyUid: Output only. The globally unique ID of the policy to be bound.
    target: Required. Immutable. Target is the full resource name of the
      resource to which the policy will be bound. Immutable once set.
    uid: Output only. The globally unique ID of the policy binding. Assigned
      when the policy binding is created.
    updateTime: Output only. The time when the policy binding was most
      recently updated.
  """

    class PolicyKindValueValuesEnum(_messages.Enum):
        """Immutable. The kind of the policy to attach in this binding: + When
    the policy is empty, this field must be set. + When the policy is set,
    this field + can be left empty and will be set to the policy kind, or +
    must set to the input policy kind

    Values:
      POLICY_KIND_UNSPECIFIED: Unspecified policy kind; Not a valid state
      PRINCIPAL_ACCESS_BOUNDARY: Principal access boundary policy kind
    """
        POLICY_KIND_UNSPECIFIED = 0
        PRINCIPAL_ACCESS_BOUNDARY = 1

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AnnotationsValue(_messages.Message):
        """Optional. User defined annotations. See
    https://google.aip.dev/148#annotations for more details such as format and
    size limitations

    Messages:
      AdditionalProperty: An additional property for a AnnotationsValue
        object.

    Fields:
      additionalProperties: Additional properties of type AnnotationsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a AnnotationsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    annotations = _messages.MessageField('AnnotationsValue', 1)
    condition = _messages.MessageField('GoogleTypeExpr', 2)
    createTime = _messages.StringField(3)
    displayName = _messages.StringField(4)
    etag = _messages.StringField(5)
    name = _messages.StringField(6)
    policy = _messages.StringField(7)
    policyKind = _messages.EnumField('PolicyKindValueValuesEnum', 8)
    policyUid = _messages.StringField(9)
    target = _messages.MessageField('GoogleIamV3betaPolicyBindingTarget', 10)
    uid = _messages.StringField(11)
    updateTime = _messages.StringField(12)