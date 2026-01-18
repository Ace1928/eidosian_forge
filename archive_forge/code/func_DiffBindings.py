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
def DiffBindings(old, new):
    """Computes the difference between two BindingsValueListEntry lists.

  Args:
    old: The original list of BindingValuesListEntry instances
    new: The updated list of BindingValuesListEntry instances

  Returns:
    A pair of BindingsTuple instances, one for roles granted between old and
      new, and one for roles removed between old and new.
  """
    tmp_old = BindingsMessageToUpdateDict(old)
    tmp_new = BindingsMessageToUpdateDict(new)
    granted = BindingsMessageToUpdateDict([])
    removed = BindingsMessageToUpdateDict([])
    for role, members in six.iteritems(tmp_old):
        removed[role].update(members.difference(tmp_new[role]))
    for role, members in six.iteritems(tmp_new):
        granted[role].update(members.difference(tmp_old[role]))
    granted = [apitools_messages.Policy.BindingsValueListEntry(role=r, members=list(m)) for r, m in six.iteritems(granted) if m]
    removed = [apitools_messages.Policy.BindingsValueListEntry(role=r, members=list(m)) for r, m in six.iteritems(removed) if m]
    return (BindingsTuple(True, granted), BindingsTuple(False, removed))