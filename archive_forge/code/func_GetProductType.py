import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def GetProductType(self):
    """Returns the PRODUCT_TYPE of this target."""
    if self._IsIosAppExtension():
        assert self._IsBundle(), 'ios_app_extension flag requires mac_bundle (target %s)' % self.spec['target_name']
        return 'com.apple.product-type.app-extension'
    if self._IsIosWatchKitExtension():
        assert self._IsBundle(), 'ios_watchkit_extension flag requires mac_bundle (target %s)' % self.spec['target_name']
        return 'com.apple.product-type.watchkit-extension'
    if self._IsIosWatchApp():
        assert self._IsBundle(), 'ios_watch_app flag requires mac_bundle (target %s)' % self.spec['target_name']
        return 'com.apple.product-type.application.watchapp'
    if self._IsXCUiTest():
        assert self._IsBundle(), 'mac_xcuitest_bundle flag requires mac_bundle (target %s)' % self.spec['target_name']
        return 'com.apple.product-type.bundle.ui-testing'
    if self._IsBundle():
        return {'executable': 'com.apple.product-type.application', 'loadable_module': 'com.apple.product-type.bundle', 'shared_library': 'com.apple.product-type.framework'}[self.spec['type']]
    else:
        return {'executable': 'com.apple.product-type.tool', 'loadable_module': 'com.apple.product-type.library.dynamic', 'shared_library': 'com.apple.product-type.library.dynamic', 'static_library': 'com.apple.product-type.library.static'}[self.spec['type']]