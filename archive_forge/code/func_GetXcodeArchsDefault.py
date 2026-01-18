import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def GetXcodeArchsDefault():
    """Returns the |XcodeArchsDefault| object to use to expand ARCHS for the
  installed version of Xcode. The default values used by Xcode for ARCHS
  and the expansion of the variables depends on the version of Xcode used.

  For all version anterior to Xcode 5.0 or posterior to Xcode 5.1 included
  uses $(ARCHS_STANDARD) if ARCHS is unset, while Xcode 5.0 to 5.0.2 uses
  $(ARCHS_STANDARD_INCLUDING_64_BIT). This variable was added to Xcode 5.0
  and deprecated with Xcode 5.1.

  For "macosx" SDKROOT, all version starting with Xcode 5.0 includes 64-bit
  architecture as part of $(ARCHS_STANDARD) and default to only building it.

  For "iphoneos" and "iphonesimulator" SDKROOT, 64-bit architectures are part
  of $(ARCHS_STANDARD_INCLUDING_64_BIT) from Xcode 5.0. From Xcode 5.1, they
  are also part of $(ARCHS_STANDARD).

  All these rules are coded in the construction of the |XcodeArchsDefault|
  object to use depending on the version of Xcode detected. The object is
  for performance reason."""
    global XCODE_ARCHS_DEFAULT_CACHE
    if XCODE_ARCHS_DEFAULT_CACHE:
        return XCODE_ARCHS_DEFAULT_CACHE
    xcode_version, _ = XcodeVersion()
    if xcode_version < '0500':
        XCODE_ARCHS_DEFAULT_CACHE = XcodeArchsDefault('$(ARCHS_STANDARD)', XcodeArchsVariableMapping(['i386']), XcodeArchsVariableMapping(['i386']), XcodeArchsVariableMapping(['armv7']))
    elif xcode_version < '0510':
        XCODE_ARCHS_DEFAULT_CACHE = XcodeArchsDefault('$(ARCHS_STANDARD_INCLUDING_64_BIT)', XcodeArchsVariableMapping(['x86_64'], ['x86_64']), XcodeArchsVariableMapping(['i386'], ['i386', 'x86_64']), XcodeArchsVariableMapping(['armv7', 'armv7s'], ['armv7', 'armv7s', 'arm64']))
    else:
        XCODE_ARCHS_DEFAULT_CACHE = XcodeArchsDefault('$(ARCHS_STANDARD)', XcodeArchsVariableMapping(['x86_64'], ['x86_64']), XcodeArchsVariableMapping(['i386', 'x86_64'], ['i386', 'x86_64']), XcodeArchsVariableMapping(['armv7', 'armv7s', 'arm64'], ['armv7', 'armv7s', 'arm64']))
    return XCODE_ARCHS_DEFAULT_CACHE