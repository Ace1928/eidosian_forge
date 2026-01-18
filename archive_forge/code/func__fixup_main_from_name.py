import os
import sys
import runpy
import types
from . import get_start_method, set_start_method
from . import process
from .context import reduction
from . import util
def _fixup_main_from_name(mod_name):
    current_main = sys.modules['__main__']
    if mod_name == '__main__' or mod_name.endswith('.__main__'):
        return
    if getattr(current_main.__spec__, 'name', None) == mod_name:
        return
    old_main_modules.append(current_main)
    main_module = types.ModuleType('__mp_main__')
    main_content = runpy.run_module(mod_name, run_name='__mp_main__', alter_sys=True)
    main_module.__dict__.update(main_content)
    sys.modules['__main__'] = sys.modules['__mp_main__'] = main_module