from typing import Dict, List, TYPE_CHECKING
import pytest
import numpy as np
import cirq
from cirq_google.engine.abstract_job import AbstractJob
class MockJob(AbstractJob):

    def engine(self) -> 'abstract_engine.AbstractEngine':
        pass

    def id(self) -> str:
        pass

    def program(self) -> 'abstract_program.AbstractProgram':
        pass

    def create_time(self) -> 'datetime.datetime':
        pass

    def update_time(self) -> 'datetime.datetime':
        pass

    def description(self) -> str:
        pass

    def set_description(self, description: str) -> 'AbstractJob':
        pass

    def labels(self) -> Dict[str, str]:
        pass

    def set_labels(self, labels: Dict[str, str]) -> 'AbstractJob':
        pass

    def add_labels(self, labels: Dict[str, str]) -> 'AbstractJob':
        pass

    def remove_labels(self, keys: List[str]) -> 'AbstractJob':
        pass

    def processor_ids(self):
        pass

    def execution_status(self):
        pass

    def failure(self):
        pass

    def get_repetitions_and_sweeps(self):
        pass

    def get_processor(self):
        pass

    def get_calibration(self):
        pass

    def cancel(self) -> None:
        pass

    def delete(self) -> None:
        pass

    async def batched_results_async(self):
        pass

    async def results_async(self):
        return [cirq.ResultDict(params={}, measurements={'a': np.asarray([t])}) for t in range(5)]

    async def calibration_results_async(self):
        pass