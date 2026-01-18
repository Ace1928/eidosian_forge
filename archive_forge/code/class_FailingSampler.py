import duet
import pytest
import cirq
class FailingSampler:

    async def run_async(self, circuit, repetitions):
        await duet.completed_future(None)
        raise Exception('job failed!')