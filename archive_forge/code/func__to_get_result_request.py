from typing import AsyncIterator, Dict, Optional, Union
import asyncio
import duet
import google.api_core.exceptions as google_exceptions
from cirq_google.cloud import quantum
from cirq_google.engine.asyncio_executor import AsyncioExecutor
def _to_get_result_request(create_program_and_job_request: quantum.QuantumRunStreamRequest) -> quantum.QuantumRunStreamRequest:
    """Converted the QuantumRunStreamRequest from a CreateQuantumProgramAndJobRequest to a
    GetQuantumResultRequest.
    """
    job = create_program_and_job_request.create_quantum_program_and_job.quantum_job
    return quantum.QuantumRunStreamRequest(parent=create_program_and_job_request.parent, get_quantum_result=quantum.GetQuantumResultRequest(parent=job.name))