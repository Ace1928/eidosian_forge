from typing import Sequence
import pytest
import duet
import numpy as np
import pandas as pd
import sympy
import cirq
class AsyncSampler(cirq.Sampler):

    async def run_sweep_async(self, program, params, repetitions: int=1):
        await duet.sleep(0.001)
        return cirq.Simulator().run_sweep(program, params, repetitions)