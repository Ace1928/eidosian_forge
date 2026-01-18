from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Union
from ..containers import PrimitiveResult
from .base_result import _BasePrimitiveResult
class BasePrimitiveJob(ABC, Generic[ResultT, StatusT]):
    """Primitive job abstract base class."""

    def __init__(self, job_id: str, **kwargs) -> None:
        """Initializes the primitive job.

        Args:
            job_id: A unique id in the context of the primitive used to run the job.
            kwargs: Any key value metadata to associate with this job.
        """
        self._job_id = job_id
        self.metadata = kwargs

    def job_id(self) -> str:
        """Return a unique id identifying the job."""
        return self._job_id

    @abstractmethod
    def result(self) -> ResultT:
        """Return the results of the job."""
        raise NotImplementedError('Subclass of BasePrimitiveJob must implement `result` method.')

    @abstractmethod
    def status(self) -> StatusT:
        """Return the status of the job."""
        raise NotImplementedError('Subclass of BasePrimitiveJob must implement `status` method.')

    @abstractmethod
    def done(self) -> bool:
        """Return whether the job has successfully run."""
        raise NotImplementedError('Subclass of BasePrimitiveJob must implement `done` method.')

    @abstractmethod
    def running(self) -> bool:
        """Return whether the job is actively running."""
        raise NotImplementedError('Subclass of BasePrimitiveJob must implement `running` method.')

    @abstractmethod
    def cancelled(self) -> bool:
        """Return whether the job has been cancelled."""
        raise NotImplementedError('Subclass of BasePrimitiveJob must implement `cancelled` method.')

    @abstractmethod
    def in_final_state(self) -> bool:
        """Return whether the job is in a final job state such as ``DONE`` or ``ERROR``."""
        raise NotImplementedError('Subclass of BasePrimitiveJob must implement `is_final_state` method.')

    @abstractmethod
    def cancel(self):
        """Attempt to cancel the job."""
        raise NotImplementedError('Subclass of BasePrimitiveJob must implement `cancel` method.')