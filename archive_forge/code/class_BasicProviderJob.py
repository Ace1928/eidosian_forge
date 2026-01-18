import warnings
from qiskit.providers import JobStatus
from qiskit.providers.job import JobV1
class BasicProviderJob(JobV1):
    """BasicProviderJob class."""
    _async = False

    def __init__(self, backend, job_id, result):
        super().__init__(backend, job_id)
        self._result = result

    def submit(self):
        """Submit the job to the backend for execution.

        Raises:
            JobError: if trying to re-submit the job.
        """
        return

    def result(self, timeout=None):
        """Get job result .

        Returns:
            qiskit.result.Result: Result object
        """
        if timeout is not None:
            warnings.warn("The timeout kwarg doesn't have any meaning with BasicProvider because execution is synchronous and the result already exists when run() returns.", UserWarning)
        return self._result

    def status(self):
        """Gets the status of the job by querying the Python's future

        Returns:
            qiskit.providers.JobStatus: The current JobStatus
        """
        return JobStatus.DONE

    def backend(self):
        """Return the instance of the backend used for this job."""
        return self._backend