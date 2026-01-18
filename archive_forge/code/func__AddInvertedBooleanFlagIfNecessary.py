from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import display_info
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.core.cache import completion_cache
def _AddInvertedBooleanFlagIfNecessary(self, added_argument, name, dest, original_kwargs):
    """Determines whether to create the --no-* flag and adds it to the parser.

    Args:
      added_argument: The argparse argument that was previously created.
      name: str, The name of the flag.
      dest: str, The dest field of the flag.
      original_kwargs: {str: object}, The original set of kwargs passed to the
        ArgumentInterceptor.

    Returns:
      The new argument that was added to the parser or None, if it was not
      necessary to create a new argument.
    """
    action = original_kwargs.get('action')
    wrapped_action = getattr(action, 'wrapped_action', None)
    if wrapped_action is not None:
        action_wrapper = action
        action = wrapped_action
    should_invert, prop = self._ShouldInvertBooleanFlag(name, action)
    if not should_invert:
        return
    default = original_kwargs.get('default', False)
    if prop:
        inverted_synopsis = bool(prop.default)
    elif default not in (True, None):
        inverted_synopsis = False
    elif default:
        inverted_synopsis = bool(default)
    else:
        inverted_synopsis = False
    kwargs = dict(original_kwargs)
    if _IsStoreTrueAction(action):
        action = 'store_false'
    elif _IsStoreFalseAction(action):
        action = 'store_true'
    if wrapped_action is not None:

        class NewAction(action_wrapper):
            pass
        NewAction.SetWrappedAction(action)
        action = NewAction
    kwargs['action'] = action
    if not kwargs.get('dest'):
        kwargs['dest'] = dest
    inverted_argument = self.parser.add_argument(name.replace('--', '--no-', 1), **kwargs)
    if inverted_synopsis:
        setattr(added_argument, 'inverted_synopsis', True)
    inverted_argument.is_hidden = True
    inverted_argument.is_required = added_argument.is_required
    return inverted_argument