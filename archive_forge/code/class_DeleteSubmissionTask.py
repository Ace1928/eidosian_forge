from s3transfer.tasks import Task
from s3transfer.tasks import SubmissionTask
class DeleteSubmissionTask(SubmissionTask):
    """Task for submitting tasks to execute an object deletion."""

    def _submit(self, client, request_executor, transfer_future, **kwargs):
        """
        :param client: The client associated with the transfer manager

        :type config: s3transfer.manager.TransferConfig
        :param config: The transfer config associated with the transfer
            manager

        :type osutil: s3transfer.utils.OSUtil
        :param osutil: The os utility associated to the transfer manager

        :type request_executor: s3transfer.futures.BoundedExecutor
        :param request_executor: The request executor associated with the
            transfer manager

        :type transfer_future: s3transfer.futures.TransferFuture
        :param transfer_future: The transfer future associated with the
            transfer request that tasks are being submitted for
        """
        call_args = transfer_future.meta.call_args
        self._transfer_coordinator.submit(request_executor, DeleteObjectTask(transfer_coordinator=self._transfer_coordinator, main_kwargs={'client': client, 'bucket': call_args.bucket, 'key': call_args.key, 'extra_args': call_args.extra_args}, is_final=True))