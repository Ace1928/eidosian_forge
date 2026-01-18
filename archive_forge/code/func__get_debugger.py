import functools
import traceback
def _get_debugger(debugger_name):
    try:
        debugger = __import__(debugger_name)
    except ImportError:
        raise ValueError("can't import %s module as a post mortem debugger" % debugger_name)
    if 'post_mortem' in dir(debugger):
        return debugger
    else:
        raise ValueError('%s is not a supported post mortem debugger' % debugger_name)