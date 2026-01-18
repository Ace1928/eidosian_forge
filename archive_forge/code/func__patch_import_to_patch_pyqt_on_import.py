from __future__ import nested_scopes
from _pydev_bundle._pydev_saved_modules import threading
import os
from _pydev_bundle import pydev_log
def _patch_import_to_patch_pyqt_on_import(patch_qt_on_import, get_qt_core_module):
    pydev_log.debug('Setting up Qt post-import monkeypatch.')
    dotted = patch_qt_on_import + '.'
    original_import = __import__
    from _pydev_bundle._pydev_sys_patch import patch_sys_module, patch_reload, cancel_patches_in_sys_module
    patch_sys_module()
    patch_reload()

    def patched_import(name, *args, **kwargs):
        if patch_qt_on_import == name or name.startswith(dotted):
            builtins.__import__ = original_import
            cancel_patches_in_sys_module()
            _internal_patch_qt(get_qt_core_module())
        return original_import(name, *args, **kwargs)
    import builtins
    builtins.__import__ = patched_import