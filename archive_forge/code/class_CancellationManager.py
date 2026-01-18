from tensorflow.python import pywrap_tfe
class CancellationManager(object):
    """A mechanism for cancelling blocking computation."""
    __slots__ = ['_impl']

    def __init__(self):
        self._impl = pywrap_tfe.TFE_NewCancellationManager()

    @property
    def is_cancelled(self):
        """Returns `True` if `CancellationManager.start_cancel` has been called."""
        return pywrap_tfe.TFE_CancellationManagerIsCancelled(self._impl)

    def start_cancel(self):
        """Cancels blocking operations that have been registered with this object."""
        pywrap_tfe.TFE_CancellationManagerStartCancel(self._impl)

    def get_cancelable_function(self, concrete_function):

        def cancellable(*args, **kwargs):
            with CancellationManagerContext(self):
                return concrete_function(*args, **kwargs)
        return cancellable