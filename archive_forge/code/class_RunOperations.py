from typing import List
class RunOperations:
    """Class that helps manage the futures of MLflow async logging."""

    def __init__(self, operation_futures):
        self._operation_futures = operation_futures or []

    def wait(self):
        """Blocks on completion of all futures."""
        from mlflow.exceptions import MlflowException
        failed_operations = []
        for future in self._operation_futures:
            try:
                future.result()
            except Exception as e:
                failed_operations.append(e)
        if len(failed_operations) > 0:
            raise MlflowException(f'The following failures occurred while performing one or more async logging operations: {failed_operations}')