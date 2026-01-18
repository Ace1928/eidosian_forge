import os
import re
import subprocess
import gyp
import gyp.common
import gyp.xcode_emulation
from gyp.common import GetEnvironFallback
import hashlib
def WriteRootHeaderSuffixRules(writer):
    extensions = sorted(COMPILABLE_EXTENSIONS.keys(), key=str.lower)
    writer.write('# Suffix rules, putting all outputs into $(obj).\n')
    for ext in extensions:
        writer.write('$(obj).$(TOOLSET)/%%.o: $(srcdir)/%%%s FORCE_DO_CMD\n' % ext)
        writer.write('\t@$(call do_cmd,%s,1)\n' % COMPILABLE_EXTENSIONS[ext])
    writer.write('\n# Try building from generated source, too.\n')
    for ext in extensions:
        writer.write('$(obj).$(TOOLSET)/%%.o: $(obj).$(TOOLSET)/%%%s FORCE_DO_CMD\n' % ext)
        writer.write('\t@$(call do_cmd,%s,1)\n' % COMPILABLE_EXTENSIONS[ext])
    writer.write('\n')
    for ext in extensions:
        writer.write('$(obj).$(TOOLSET)/%%.o: $(obj)/%%%s FORCE_DO_CMD\n' % ext)
        writer.write('\t@$(call do_cmd,%s,1)\n' % COMPILABLE_EXTENSIONS[ext])
    writer.write('\n')