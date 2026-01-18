import datetime
import pytest
from google.protobuf.timestamp_pb2 import Timestamp
from cirq_google.cloud import quantum
from cirq_google.engine.abstract_local_processor import AbstractLocalProcessor
class NothingProcessor(AbstractLocalProcessor):
    """A processor for testing."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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

    def list_programs(self, *args, **kwargs):
        pass

    def get_program(self, *args, **kwargs):
        pass