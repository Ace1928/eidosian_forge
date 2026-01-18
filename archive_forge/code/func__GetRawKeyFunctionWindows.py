from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import sys
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import platforms
def _GetRawKeyFunctionWindows():
    """_GetRawKeyFunction helper using Windows APIs."""
    import msvcrt

    def _GetRawKeyWindows():
        """Reads and returns one keypress from stdin, no echo, using Windows APIs.

    Returns:
      The key name, None for EOF, <*> for function keys, otherwise a
      character.
    """
        windows_to_key = {'H': '<UP-ARROW>', 'P': '<DOWN-ARROW>', 'K': '<LEFT-ARROW>', 'M': '<RIGHT-ARROW>', 'I': '<PAGE-UP>', 'Q': '<PAGE-DOWN>', 'G': '<HOME>', 'O': '<END>'}
        sys.stdout.flush()

        def _GetKeyChar():
            return encoding.Decode(msvcrt.getch())
        c = _GetKeyChar()
        if c in (_WINDOWS_CSI_1, _WINDOWS_CSI_2):
            return windows_to_key.get(_GetKeyChar(), '')
        return None if c in (_CONTROL_D, _CONTROL_Z) else c
    return _GetRawKeyWindows