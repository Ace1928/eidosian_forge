import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def GetInstallName(self):
    """Return LD_DYLIB_INSTALL_NAME for this target."""
    if self.spec['type'] != 'shared_library' and (self.spec['type'] != 'loadable_module' or self._IsBundle()):
        return None
    default_install_name = '$(DYLIB_INSTALL_NAME_BASE:standardizepath)/$(EXECUTABLE_PATH)'
    install_name = self.GetPerTargetSetting('LD_DYLIB_INSTALL_NAME', default=default_install_name)
    if '$' in install_name:
        assert install_name in ('$(DYLIB_INSTALL_NAME_BASE:standardizepath)/$(WRAPPER_NAME)/$(PRODUCT_NAME)', default_install_name), "Variables in LD_DYLIB_INSTALL_NAME are not generally supported yet in target '%s' (got '%s')" % (self.spec['target_name'], install_name)
        install_name = install_name.replace('$(DYLIB_INSTALL_NAME_BASE:standardizepath)', self._StandardizePath(self.GetInstallNameBase()))
        if self._IsBundle():
            install_name = install_name.replace('$(WRAPPER_NAME)', self.GetWrapperName())
            install_name = install_name.replace('$(PRODUCT_NAME)', self.GetProductName())
        else:
            assert '$(WRAPPER_NAME)' not in install_name
            assert '$(PRODUCT_NAME)' not in install_name
        install_name = install_name.replace('$(EXECUTABLE_PATH)', self.GetExecutablePath())
    return install_name