import os
import re
import subprocess
import gyp
import gyp.common
import gyp.xcode_emulation
from gyp.common import GetEnvironFallback
import hashlib
def WriteMacInfoPlist(self, bundle_deps):
    """Write Makefile code for bundle Info.plist files."""
    info_plist, out, defines, extra_env = gyp.xcode_emulation.GetMacInfoPlist(generator_default_variables['PRODUCT_DIR'], self.xcode_settings, lambda p: Sourceify(self.Absolutify(p)))
    if not info_plist:
        return
    if defines:
        intermediate_plist = '$(obj).$(TOOLSET)/$(TARGET)/' + os.path.basename(info_plist)
        self.WriteList(defines, intermediate_plist + ': INFOPLIST_DEFINES', '-D', quoter=EscapeCppDefine)
        self.WriteMakeRule([intermediate_plist], [info_plist], ['$(call do_cmd,infoplist)', '@plutil -convert xml1 $@ $@'])
        info_plist = intermediate_plist
    self.WriteSortedXcodeEnv(out, self.GetSortedXcodeEnv(additional_settings=extra_env))
    self.WriteDoCmd([out], [info_plist], 'mac_tool,,,copy-info-plist', part_of_all=True)
    bundle_deps.append(out)