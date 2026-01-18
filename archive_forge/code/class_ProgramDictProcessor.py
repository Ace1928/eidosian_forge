import datetime
from typing import Dict, Optional, Union
import pytest
import cirq
import cirq_google
import sympy
import numpy as np
from cirq_google.api import v2
from cirq_google.engine.abstract_local_job_test import NothingJob
from cirq_google.engine.abstract_local_program_test import NothingProgram
from cirq_google.engine.abstract_local_processor import AbstractLocalProcessor
from cirq_google.engine.abstract_program import AbstractProgram
from cirq_google.engine.simulated_local_engine import SimulatedLocalEngine
from cirq_google.engine.simulated_local_processor import SimulatedLocalProcessor
class ProgramDictProcessor(AbstractLocalProcessor):
    """A processor that has a dictionary of programs for testing."""

    def __init__(self, programs: Dict[str, AbstractProgram], **kwargs):
        super().__init__(**kwargs)
        self._programs = programs

    def get_calibration(self, *args, **kwargs):
        pass

    def get_latest_calibration(self, *args, **kwargs):
        pass

    def get_current_calibration(self, *args, **kwargs):
        pass

    def get_device(self, *args, **kwargs):
        pass

    def get_device_specification(self, *args, **kwargs):
        pass

    def health(self, *args, **kwargs):
        pass

    def list_calibrations(self, *args, **kwargs):
        pass

    async def run_batch_async(self, *args, **kwargs):
        pass

    async def run_calibration_async(self, *args, **kwargs):
        pass

    async def run_sweep_async(self, *args, **kwargs):
        pass

    def get_sampler(self, *args, **kwargs):
        pass

    def supported_languages(self, *args, **kwargs):
        pass

    def list_programs(self, created_before: Optional[Union[datetime.datetime, datetime.date]]=None, created_after: Optional[Union[datetime.datetime, datetime.date]]=None, has_labels: Optional[Dict[str, str]]=None):
        """Lists all programs regardless of filters.

        This isn't really correct, but we don't want to test test functionality."""
        return self._programs.values()

    def get_program(self, program_id: str) -> AbstractProgram:
        return self._programs[program_id]