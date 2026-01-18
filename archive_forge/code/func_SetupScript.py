import errno
import os
import re
import subprocess
import sys
import glob
def SetupScript(self, target_arch):
    script_data = self._SetupScriptInternal(target_arch)
    script_path = script_data[0]
    if not os.path.exists(script_path):
        raise Exception('%s is missing - make sure VC++ tools are installed.' % script_path)
    return script_data