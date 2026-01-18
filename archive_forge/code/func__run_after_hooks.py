import abc
import inspect
from stevedore import extension
from cliff import _argparse
def _run_after_hooks(self, parsed_args, return_code):
    """Calls after() method of the hooks.

        This method is intended to be called from the run() method after
        take_action() is called.

        This method should only be overridden by developers creating new
        command base classes and only if it is necessary to have different
        hook processing behavior.
        """
    for hook in self._hooks:
        ret = hook.obj.after(parsed_args, return_code)
        if ret is not None:
            return_code = ret
    return return_code