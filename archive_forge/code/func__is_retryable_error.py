from typing import AsyncIterator, Dict, Optional, Union
import asyncio
import duet
import google.api_core.exceptions as google_exceptions
from cirq_google.cloud import quantum
from cirq_google.engine.asyncio_executor import AsyncioExecutor
def _is_retryable_error(e: google_exceptions.GoogleAPICallError) -> bool:
    return any((isinstance(e, exception_type) for exception_type in RETRYABLE_GOOGLE_API_EXCEPTIONS))