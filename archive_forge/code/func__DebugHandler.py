import google.apputils.debug
import sys
import gflags as flags
def _DebugHandler(exc_class, value, tb):
    if not flags.FLAGS.pdb or hasattr(sys, 'ps1') or (not sys.stderr.isatty()):
        old_excepthook(exc_class, value, tb)
    else:
        import traceback
        import pdb
        traceback.print_exception(exc_class, value, tb)
        print
        pdb.pm()