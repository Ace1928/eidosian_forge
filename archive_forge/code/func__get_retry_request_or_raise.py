from typing import AsyncIterator, Dict, Optional, Union
import asyncio
import duet
import google.api_core.exceptions as google_exceptions
from cirq_google.cloud import quantum
from cirq_google.engine.asyncio_executor import AsyncioExecutor
def _get_retry_request_or_raise(error: quantum.StreamError, current_request: quantum.QuantumRunStreamRequest, create_program_and_job_request: quantum.QuantumRunStreamRequest, create_job_request: quantum.QuantumRunStreamRequest, get_result_request: quantum.QuantumRunStreamRequest):
    """Decide whether the given stream error is retryable.

    If it is, returns the next stream request to send upon retry. Otherwise, raises an error.
    """
    if error.code == Code.PROGRAM_DOES_NOT_EXIST:
        if 'create_quantum_job' in current_request:
            return create_program_and_job_request
    elif error.code == Code.PROGRAM_ALREADY_EXISTS:
        if 'create_quantum_program_and_job' in current_request:
            return get_result_request
    elif error.code == Code.JOB_DOES_NOT_EXIST:
        if 'get_quantum_result' in current_request:
            return create_job_request
    elif error.code == Code.JOB_ALREADY_EXISTS:
        if not 'get_quantum_result' in current_request:
            return get_result_request
    raise StreamError(error.message)