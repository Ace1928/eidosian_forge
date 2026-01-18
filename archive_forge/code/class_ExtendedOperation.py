import threading
from google.api_core import exceptions
from google.api_core.future import polling
class ExtendedOperation(polling.PollingFuture):
    """An ExtendedOperation future for interacting with a Google API Long-Running Operation.

    Args:
        extended_operation (proto.Message): The initial operation.
        refresh (Callable[[], type(extended_operation)]): A callable that returns
            the latest state of the operation.
        cancel (Callable[[], None]): A callable that tries to cancel the operation.
        polling Optional(google.api_core.retry.Retry): The configuration used
            for polling. This can be used to control how often :meth:`done`
            is polled. If the ``timeout`` argument to :meth:`result` is
            specified it will override the ``polling.timeout`` property.
        retry Optional(google.api_core.retry.Retry): DEPRECATED use ``polling``
            instead. If specified it will override ``polling`` parameter to
            maintain backward compatibility.

    Note: Most long-running API methods use google.api_core.operation.Operation
    This class is a wrapper for a subset of methods that use alternative
    Long-Running Operation (LRO) semantics.

    Note: there is not a concrete type the extended operation must be.
    It MUST have fields that correspond to the following, POSSIBLY WITH DIFFERENT NAMES:
    * name: str
    * status: Union[str, bool, enum.Enum]
    * error_code: int
    * error_message: str
    """

    def __init__(self, extended_operation, refresh, cancel, polling=polling.DEFAULT_POLLING, **kwargs):
        super().__init__(polling=polling, **kwargs)
        self._extended_operation = extended_operation
        self._refresh = refresh
        self._cancel = cancel
        self._cancelled = False
        self._completion_lock = threading.Lock()
        self._handle_refreshed_operation()

    @property
    def name(self):
        return self._extended_operation.name

    @property
    def status(self):
        return self._extended_operation.status

    @property
    def error_code(self):
        return self._extended_operation.error_code

    @property
    def error_message(self):
        return self._extended_operation.error_message

    def __getattr__(self, name):
        return getattr(self._extended_operation, name)

    def done(self, retry=None):
        self._refresh_and_update(retry)
        return self._extended_operation.done

    def cancel(self):
        if self.done():
            return False
        self._cancel()
        self._cancelled = True
        return True

    def cancelled(self):
        if not self._cancelled:
            return False
        self._refresh_and_update()
        return self._extended_operation.done

    def _refresh_and_update(self, retry=None):
        if not self._extended_operation.done:
            self._extended_operation = self._refresh(retry=retry) if retry else self._refresh()
            self._handle_refreshed_operation()

    def _handle_refreshed_operation(self):
        with self._completion_lock:
            if not self._extended_operation.done:
                return
            if self.error_code and self.error_message:
                errors = []
                if hasattr(self, 'error') and hasattr(self.error, 'errors'):
                    errors = self.error.errors
                exception = exceptions.from_http_status(status_code=self.error_code, message=self.error_message, response=self._extended_operation, errors=errors)
                self.set_exception(exception)
            elif self.error_code or self.error_message:
                exception = exceptions.GoogleAPICallError(f'Unexpected error {self.error_code}: {self.error_message}')
                self.set_exception(exception)
            else:
                self.set_result(None)

    @classmethod
    def make(cls, refresh, cancel, extended_operation, **kwargs):
        """
        Return an instantiated ExtendedOperation (or child) that wraps
        * a refresh callable
        * a cancel callable (can be a no-op)
        * an initial result

        .. note::
            It is the caller's responsibility to set up refresh and cancel
            with their correct request argument.
            The reason for this is that the services that use Extended Operations
            have rpcs that look something like the following:

            // service.proto
            service MyLongService {
                rpc StartLongTask(StartLongTaskRequest) returns (ExtendedOperation) {
                    option (google.cloud.operation_service) = "CustomOperationService";
                }
            }

            service CustomOperationService {
                rpc Get(GetOperationRequest) returns (ExtendedOperation) {
                    option (google.cloud.operation_polling_method) = true;
                }
            }

            Any info needed for the poll, e.g. a name, path params, etc.
            is held in the request, which the initial client method is in a much
            better position to make made because the caller made the initial request.

            TL;DR: the caller sets up closures for refresh and cancel that carry
            the properly configured requests.

        Args:
            refresh (Callable[Optional[Retry]][type(extended_operation)]): A callable that
                returns the latest state of the operation.
            cancel (Callable[][Any]): A callable that tries to cancel the operation
                on a best effort basis.
            extended_operation (Any): The initial response of the long running method.
                See the docstring for ExtendedOperation.__init__ for requirements on
                the type and fields of extended_operation
        """
        return cls(extended_operation, refresh, cancel, **kwargs)