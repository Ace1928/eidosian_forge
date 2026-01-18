from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CheckSet(_messages.Message):
    """A conjunction of policy checks, scoped to a particular namespace or
  Kubernetes service account. In order for evaluation of a `CheckSet` to
  return "allowed" for a given image in a given Pod, one of the following
  conditions must be satisfied: * The image is explicitly exempted by an entry
  in `image_allowlist`, OR * ALL of the `checks` evaluate to "allowed".

  Fields:
    checks: Optional. The checks to apply. The ultimate result of evaluating
      the check set will be "allow" if and only if every check in `checks`
      evaluates to "allow". If `checks` is empty, the default behavior is
      "always allow".
    displayName: Optional. A user-provided name for this `CheckSet`. This
      field has no effect on the policy evaluation behavior except to improve
      readability of messages in evaluation results.
    imageAllowlist: Optional. Images exempted from this `CheckSet`. If any of
      the patterns match the image being evaluated, no checks in the
      `CheckSet` will be evaluated.
    scope: Optional. The scope to which this `CheckSet` applies. If unset or
      an empty string (the default), applies to all namespaces and service
      accounts. See the `Scope` message documentation for details on scoping
      rules.
  """
    checks = _messages.MessageField('Check', 1, repeated=True)
    displayName = _messages.StringField(2)
    imageAllowlist = _messages.MessageField('ImageAllowlist', 3)
    scope = _messages.MessageField('Scope', 4)