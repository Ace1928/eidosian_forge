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
def BindingStringToTuple(is_grant, input_str):
    """Parses an iam ch bind string to a list of binding tuples.

  Args:
    is_grant: If true, binding is to be appended to IAM policy; else, delete
              this binding from the policy.
    input_str: A string representing a member-role binding.
               e.g. user:foo@bar.com:objectAdmin
                    user:foo@bar.com:objectAdmin,objectViewer
                    user:foo@bar.com
                    allUsers
                    deleted:user:foo@bar.com?uid=123:objectAdmin,objectViewer
                    deleted:serviceAccount:foo@bar.com?uid=123

  Raises:
    CommandException in the case of invalid input.

  Returns:
    A BindingsDictTuple instance.
  """
    if not input_str.count(':'):
        input_str += ':'
    tokens = input_str.split(':')
    public_members = {s.lower(): s for s in PUBLIC_MEMBERS}
    types = {s.lower(): s for s in TYPES}
    discouraged_types = {s.lower(): s for s in DISCOURAGED_TYPES}
    possible_public_member_or_type = tokens[0].lower()
    possible_type = '%s:%s' % (tokens[0].lower(), tokens[1].lower())
    if possible_public_member_or_type in public_members:
        tokens[0] = public_members[possible_public_member_or_type]
    elif possible_public_member_or_type in types:
        tokens[0] = types[possible_public_member_or_type]
    elif possible_public_member_or_type in discouraged_types:
        tokens[0] = discouraged_types[possible_public_member_or_type]
    elif possible_type in types:
        tokens[0], tokens[1] = types[possible_type].split(':')
    input_str = ':'.join(tokens)
    removing_discouraged_type = not is_grant and tokens[0] in DISCOURAGED_TYPES
    if input_str.count(':') == 1:
        if '%s:%s' % (tokens[0], tokens[1]) in TYPES:
            raise CommandException('Incorrect public member type for binding %s' % input_str)
        elif tokens[0] in PUBLIC_MEMBERS:
            member, roles = tokens
        elif tokens[0] in TYPES or removing_discouraged_type:
            member = input_str
            roles = DROP_ALL
        else:
            raise CommandException('Incorrect public member type for binding %s' % input_str)
    elif input_str.count(':') == 2:
        if '%s:%s' % (tokens[0], tokens[1]) in TYPES:
            member = input_str
            roles = DROP_ALL
        elif removing_discouraged_type:
            member_type, project_id, roles = tokens
            member = '%s:%s' % (member_type, project_id)
        else:
            member_type, member_id, roles = tokens
            _check_member_type(member_type, input_str)
            member = '%s:%s' % (member_type, member_id)
    elif input_str.count(':') == 3:
        member_type_p1, member_type_p2, member_id, roles = input_str.split(':')
        member_type = '%s:%s' % (member_type_p1, member_type_p2)
        _check_member_type(member_type, input_str)
        member = '%s:%s' % (member_type, member_id)
    else:
        raise CommandException('Invalid ch format %s' % input_str)
    if is_grant and (not roles):
        raise CommandException('Must specify a role to grant.')
    roles = [ResolveRole(r) for r in roles.split(',')]
    bindings = [{'role': r, 'members': [member]} for r in set(roles)]
    return BindingsDictTuple(is_grant=is_grant, bindings=bindings)