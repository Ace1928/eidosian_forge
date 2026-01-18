from typing import Optional, TYPE_CHECKING
from cirq_google.engine.abstract_local_program import AbstractLocalProgram
from cirq_google.engine.local_simulation_type import LocalSimulationType
class SimulatedLocalProgram(AbstractLocalProgram):
    """A program backed by a (local) sampler.

    This class functions as a parent class for a `SimulatedLocalJob`
    object.
    """

    def __init__(self, *args, program_id: str, simulation_type: LocalSimulationType=LocalSimulationType.SYNCHRONOUS, processor: Optional['SimulatedLocalProcessor']=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._id = program_id
        self._processor = processor

    def delete(self, delete_jobs: bool=False) -> None:
        if self._processor:
            self._processor.remove_program(self._id)
        if delete_jobs:
            for job in list(self._jobs.values()):
                job.delete()

    def delete_job(self, job_id: str) -> None:
        del self._jobs[job_id]

    def id(self) -> str:
        return self._id