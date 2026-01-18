import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def _GetIOSPostbuilds(self, configname, output_binary):
    """Return a shell command to codesign the iOS output binary so it can
    be deployed to a device.  This should be run as the very last step of the
    build."""
    if not (self.isIOS and (self.spec['type'] == 'executable' or self._IsXCTest()) or self.IsIosFramework()):
        return []
    postbuilds = []
    product_name = self.GetFullProductName()
    settings = self.xcode_settings[configname]
    if self._IsXCTest():
        source = os.path.join('${BUILT_PRODUCTS_DIR}', product_name)
        test_host = os.path.dirname(settings.get('TEST_HOST'))
        xctest_destination = os.path.join(test_host, 'PlugIns', product_name)
        postbuilds.extend([f'ditto {source} {xctest_destination}'])
    key = self._GetIOSCodeSignIdentityKey(settings)
    if not key:
        return postbuilds
    unimpl = ['OTHER_CODE_SIGN_FLAGS']
    unimpl = set(unimpl) & set(self.xcode_settings[configname].keys())
    if unimpl:
        print('Warning: Some codesign keys not implemented, ignoring: %s' % ', '.join(sorted(unimpl)))
    if self._IsXCTest():
        test_host = os.path.dirname(settings.get('TEST_HOST'))
        frameworks_dir = os.path.join(test_host, 'Frameworks')
        platform_root = self._XcodePlatformPath(configname)
        frameworks = ['Developer/Library/PrivateFrameworks/IDEBundleInjection.framework', 'Developer/Library/Frameworks/XCTest.framework']
        for framework in frameworks:
            source = os.path.join(platform_root, framework)
            destination = os.path.join(frameworks_dir, os.path.basename(framework))
            postbuilds.extend([f'ditto {source} {destination}'])
            postbuilds.extend(['%s code-sign-bundle "%s" "%s" "%s" "%s" %s' % (os.path.join('${TARGET_BUILD_DIR}', 'gyp-mac-tool'), key, settings.get('CODE_SIGN_ENTITLEMENTS', ''), settings.get('PROVISIONING_PROFILE', ''), destination, True)])
        plugin_dir = os.path.join(test_host, 'PlugIns')
        targets = [os.path.join(plugin_dir, product_name), test_host]
        for target in targets:
            postbuilds.extend(['%s code-sign-bundle "%s" "%s" "%s" "%s" %s' % (os.path.join('${TARGET_BUILD_DIR}', 'gyp-mac-tool'), key, settings.get('CODE_SIGN_ENTITLEMENTS', ''), settings.get('PROVISIONING_PROFILE', ''), target, True)])
    postbuilds.extend(['%s code-sign-bundle "%s" "%s" "%s" "%s" %s' % (os.path.join('${TARGET_BUILD_DIR}', 'gyp-mac-tool'), key, settings.get('CODE_SIGN_ENTITLEMENTS', ''), settings.get('PROVISIONING_PROFILE', ''), os.path.join('${BUILT_PRODUCTS_DIR}', product_name), False)])
    return postbuilds