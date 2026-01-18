from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OSPolicyResource(_messages.Message):
    """An OS policy resource is used to define the desired state configuration
  and provides a specific functionality like installing/removing packages,
  executing a script etc. The system ensures that resources are always in
  their desired state by taking necessary actions if they have drifted from
  their desired state.

  Fields:
    exec_: Exec resource
    file: File resource
    id: Required. The id of the resource with the following restrictions: *
      Must contain only lowercase letters, numbers, and hyphens. * Must start
      with a letter. * Must be between 1-63 characters. * Must end with a
      number or a letter. * Must be unique within the OS policy.
    pkg: Package resource
    repository: Package repository resource
  """
    exec_ = _messages.MessageField('OSPolicyResourceExecResource', 1)
    file = _messages.MessageField('OSPolicyResourceFileResource', 2)
    id = _messages.StringField(3)
    pkg = _messages.MessageField('OSPolicyResourcePackageResource', 4)
    repository = _messages.MessageField('OSPolicyResourceRepositoryResource', 5)