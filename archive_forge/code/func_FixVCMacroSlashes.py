import re
import sys
def FixVCMacroSlashes(s):
    """Replace macros which have excessive following slashes.

  These macros are known to have a built-in trailing slash. Furthermore, many
  scripts hiccup on processing paths with extra slashes in the middle.

  This list is probably not exhaustive.  Add as needed.
  """
    if '$' in s:
        s = fix_vc_macro_slashes_regex.sub('\\1', s)
    return s