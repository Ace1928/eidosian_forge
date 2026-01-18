import os
import warnings
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from threading import get_ident as get_current_thread_id
import mlflow
from mlflow.utils import logging_utils
def _modify_patch_state_if_necessary(self):
    """
        Patches or unpatches `warnings.showwarning` if necessary, as determined by:
            - Whether or not `warnings.showwarning` is already patched
            - Whether or not any custom warning state has been configured on the warnings
              controller (i.e. disablement or rerouting of certain warnings globally or for a
              particular thread)

        Note that reassigning `warnings.showwarning` is the standard / recommended approach for
        modifying warning message display behaviors. For reference, see
        https://docs.python.org/3/library/warnings.html#warnings.showwarning
        """
    with self._state_lock:
        if self._should_patch_showwarning() and (not self._did_patch_showwarning):
            self._original_showwarning = warnings.showwarning
            warnings.showwarning = self._patched_showwarning
            self._did_patch_showwarning = True
        elif not self._should_patch_showwarning() and self._did_patch_showwarning:
            warnings.showwarning = self._original_showwarning
            self._did_patch_showwarning = False