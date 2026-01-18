import threading
from google.api_core import exceptions
from google.api_core.future import polling

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
        