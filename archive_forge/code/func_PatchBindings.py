from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from collections import defaultdict
from collections import namedtuple
import six
from apitools.base.protorpclite import protojson
from gslib.exception import CommandException
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
def PatchBindings(base, diff, is_grant):
    """Patches a diff list of BindingsValueListEntry to the base.

  Will remove duplicate members for any given role on a grant operation.

  Args:
    base (dict): A dictionary returned by BindingsMessageToUpdateDict or
      BindingsDictToUpdateDict representing a resource's current
      IAM policy.
    diff (dict): A dictionary returned by BindingsMessageToUpdateDict or
      BindingsDictToUpdateDict representing the IAM policy bindings to
      add/remove from `base`.
    is_grant (bool): True if `diff` should be added to `base`, False
      if it should be removed from `base`.

  Returns:
    A {role: set(members)} dictionary created by applying `diff` to `base`.
  """
    if is_grant:
        for role, members in six.iteritems(diff):
            if not role:
                raise CommandException('Role must be specified for a grant request.')
            base[role].update(members)
    else:
        for role in base:
            base[role].difference_update(diff[role])
            base[role].difference_update(diff[DROP_ALL])
    return {role: members for role, members in six.iteritems(base) if members}