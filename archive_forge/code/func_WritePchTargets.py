import os
import re
import subprocess
import gyp
import gyp.common
import gyp.xcode_emulation
from gyp.common import GetEnvironFallback
import hashlib
def WritePchTargets(self, pch_commands):
    """Writes make rules to compile prefix headers."""
    if not pch_commands:
        return
    for gch, lang_flag, lang, input in pch_commands:
        extra_flags = {'c': '$(CFLAGS_C_$(BUILDTYPE))', 'cc': '$(CFLAGS_CC_$(BUILDTYPE))', 'm': '$(CFLAGS_C_$(BUILDTYPE)) $(CFLAGS_OBJC_$(BUILDTYPE))', 'mm': '$(CFLAGS_CC_$(BUILDTYPE)) $(CFLAGS_OBJCC_$(BUILDTYPE))'}[lang]
        var_name = {'c': 'GYP_PCH_CFLAGS', 'cc': 'GYP_PCH_CXXFLAGS', 'm': 'GYP_PCH_OBJCFLAGS', 'mm': 'GYP_PCH_OBJCXXFLAGS'}[lang]
        self.WriteLn(f'{gch}: {var_name} := {lang_flag} ' + '$(DEFS_$(BUILDTYPE)) $(INCS_$(BUILDTYPE)) $(CFLAGS_$(BUILDTYPE)) ' + extra_flags)
        self.WriteLn(f'{gch}: {input} FORCE_DO_CMD')
        self.WriteLn('\t@$(call do_cmd,pch_%s,1)' % lang)
        self.WriteLn('')
        assert ' ' not in gch, 'Spaces in gch filenames not supported (%s)' % gch
        self.WriteLn('all_deps += %s' % gch)
        self.WriteLn('')