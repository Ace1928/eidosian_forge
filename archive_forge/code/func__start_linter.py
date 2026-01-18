import sys
from os.path import join, dirname, abspath, isdir
def _start_linter():
    """
    This is a pre-alpha API. You're not supposed to use it at all, except for
    testing. It will very likely change.
    """
    import jedi
    if '--debug' in sys.argv:
        jedi.set_debug_function()
    for path in sys.argv[2:]:
        if path.startswith('--'):
            continue
        if isdir(path):
            import fnmatch
            import os
            paths = []
            for root, dirnames, filenames in os.walk(path):
                for filename in fnmatch.filter(filenames, '*.py'):
                    paths.append(os.path.join(root, filename))
        else:
            paths = [path]
        try:
            for p in paths:
                for error in jedi.Script(path=p)._analysis():
                    print(error)
        except Exception:
            if '--pdb' in sys.argv:
                import traceback
                traceback.print_exc()
                import pdb
                pdb.post_mortem()
            else:
                raise