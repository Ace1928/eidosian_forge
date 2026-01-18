from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions
def ValidateSetPasswordRequest(args):
    """Validates that the arguments for setting a password are correct.

  Args:
    args: argparse.Namespace, The arguments that this command was invoked with.

  Returns:
    throws exception or None
  """
    if hasattr(args, 'retain_password') and args.retain_password and (not args.password):
        raise exceptions.InvalidArgumentException('--retain-password', 'Must set --password to non-empty value.')
    if hasattr(args, 'discard_dual_password') and args.discard_dual_password and args.password:
        raise exceptions.InvalidArgumentException('--discard-dual-password', 'Cannot set --password to non-empty value ' + 'while discarding the old password.')