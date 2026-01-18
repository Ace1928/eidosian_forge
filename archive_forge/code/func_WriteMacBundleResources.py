import os
import re
import subprocess
import gyp
import gyp.common
import gyp.xcode_emulation
from gyp.common import GetEnvironFallback
import hashlib
def WriteMacBundleResources(self, resources, bundle_deps):
    """Writes Makefile code for 'mac_bundle_resources'."""
    self.WriteLn('### Generated for mac_bundle_resources')
    for output, res in gyp.xcode_emulation.GetMacBundleResources(generator_default_variables['PRODUCT_DIR'], self.xcode_settings, [Sourceify(self.Absolutify(r)) for r in resources]):
        _, ext = os.path.splitext(output)
        if ext != '.xcassets':
            self.WriteDoCmd([output], [res], 'mac_tool,,,copy-bundle-resource', part_of_all=True)
            bundle_deps.append(output)