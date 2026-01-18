from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from typing import Dict, Union
from googlecloudsdk.calliope import parser_extensions
def FlagToUpdateMaskPaths(message_to_flags: Dict[str, Level]) -> Dict[str, str]:
    """Construct flag to update mask paths during runtime.

  From top level field, combine the string up to the leave dict.

  Flag fields are unique.

  Args:
    message_to_flags: Receive value from MESSAGE_TO_FLAGS.

  Returns:
    A dictionary that maps each flag to the corresponding field update path.

  Given the below message to flag structure,

    {
      parent: {
        child1: {
          foo: '--flag-for-foo',
          child2: {
            bar: '--flag-for-bar'
          }
        }
      }
    }

  It should produce the following flag to update path mapping:

    {
      '--flag-for-foo': parent.child1.foo,
      '--flag-for-bar': parent.child1.child2.bar,
    }
  """

    def Recursive(level: Level) -> Dict[str, str]:
        ret = {}
        for curr_path, flag_or_level in level.items():
            if isinstance(flag_or_level, str):
                ret[flag_or_level] = curr_path
            else:
                for key, remain_path in Recursive(flag_or_level).items():
                    ret[key] = curr_path + '.' + remain_path
        else:
            return ret
    return Recursive(message_to_flags)