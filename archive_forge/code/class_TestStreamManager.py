from typing import AsyncIterable, AsyncIterator, Awaitable, List, Sequence, Union
import asyncio
import concurrent
from unittest import mock
import duet
import pytest
import google.api_core.exceptions as google_exceptions
from cirq_google.engine.asyncio_executor import AsyncioExecutor
from cirq_google.engine.stream_manager import (
from cirq_google.cloud import quantum
class TestStreamManager:

    @mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
    def test_submit_expects_result_response(self, client_constructor):
        expected_result = quantum.QuantumResult(parent='projects/proj/programs/prog/jobs/job0')
        fake_client, manager = setup(client_constructor)

        async def test():
            async with duet.timeout_scope(5):
                actual_result_future = manager.submit(REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB0)
                await fake_client.wait_for_requests()
                await fake_client.reply(quantum.QuantumRunStreamResponse(result=expected_result))
                actual_result = await actual_result_future
                manager.stop()
                assert actual_result == expected_result
                assert len(fake_client.all_stream_requests) == 1
                assert 'create_quantum_program_and_job' in fake_client.all_stream_requests[0]
        duet.run(test)

    @mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
    def test_submit_program_without_name_raises(self, client_constructor):
        _, manager = setup(client_constructor)

        async def test():
            async with duet.timeout_scope(5):
                with pytest.raises(ValueError, match='Program name must be set'):
                    await manager.submit(REQUEST_PROJECT_NAME, quantum.QuantumProgram(), REQUEST_JOB0)
                manager.stop()
        duet.run(test)

    @mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
    def test_submit_cancel_future_expects_engine_cancellation_rpc_call(self, client_constructor):
        fake_client, manager = setup(client_constructor)

        async def test():
            async with duet.timeout_scope(5):
                result_future = manager.submit(REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB0)
                result_future.cancel()
                await duet.sleep(1)
                manager.stop()
                assert len(fake_client.all_cancel_requests) == 1
                assert fake_client.all_cancel_requests[0] == quantum.CancelQuantumJobRequest(name='projects/proj/programs/prog/jobs/job0')
        duet.run(test)

    @mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
    def test_submit_stream_broken_twice_expects_retry_with_get_quantum_result_twice(self, client_constructor):
        expected_result = quantum.QuantumResult(parent='projects/proj/programs/prog/jobs/job0')
        fake_client, manager = setup(client_constructor)

        async def test():
            async with duet.timeout_scope(5):
                actual_result_future = manager.submit(REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB0)
                await fake_client.wait_for_requests()
                await fake_client.reply(google_exceptions.ServiceUnavailable('unavailable'))
                await fake_client.wait_for_requests()
                await fake_client.reply(google_exceptions.ServiceUnavailable('unavailable'))
                await fake_client.wait_for_requests()
                await fake_client.reply(quantum.QuantumRunStreamResponse(result=expected_result))
                actual_result = await actual_result_future
                manager.stop()
                assert actual_result == expected_result
                assert len(fake_client.all_stream_requests) == 3
                assert 'create_quantum_program_and_job' in fake_client.all_stream_requests[0]
                assert 'get_quantum_result' in fake_client.all_stream_requests[1]
                assert 'get_quantum_result' in fake_client.all_stream_requests[2]
        duet.run(test)

    @pytest.mark.parametrize('error', [google_exceptions.InternalServerError('server error'), google_exceptions.ServiceUnavailable('unavailable')])
    @mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
    def test_submit_with_retryable_stream_breakage_expects_get_result_request(self, client_constructor, error):
        expected_result = quantum.QuantumResult(parent='projects/proj/programs/prog/jobs/job0')
        fake_client, manager = setup(client_constructor)

        async def test():
            async with duet.timeout_scope(5):
                actual_result_future = manager.submit(REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB0)
                await fake_client.wait_for_requests()
                await fake_client.reply(error)
                await fake_client.wait_for_requests()
                await fake_client.reply(quantum.QuantumRunStreamResponse(result=expected_result))
                await actual_result_future
                manager.stop()
                assert len(fake_client.all_stream_requests) == 2
                assert 'create_quantum_program_and_job' in fake_client.all_stream_requests[0]
                assert 'get_quantum_result' in fake_client.all_stream_requests[1]
        duet.run(test)

    @pytest.mark.parametrize('error', [google_exceptions.DeadlineExceeded('deadline exceeded'), google_exceptions.FailedPrecondition('failed precondition'), google_exceptions.Forbidden('forbidden'), google_exceptions.InvalidArgument('invalid argument'), google_exceptions.ResourceExhausted('resource exhausted'), google_exceptions.TooManyRequests('too many requests'), google_exceptions.Unauthenticated('unauthenticated'), google_exceptions.Unauthorized('unauthorized'), google_exceptions.Unknown('unknown')])
    @mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
    def test_submit_with_non_retryable_stream_breakage_raises_error(self, client_constructor, error):
        fake_client, manager = setup(client_constructor)

        async def test():
            async with duet.timeout_scope(5):
                actual_result_future = manager.submit(REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB0)
                await fake_client.wait_for_requests()
                await fake_client.reply(error)
                with pytest.raises(type(error)):
                    await actual_result_future
                manager.stop()
                assert len(fake_client.all_stream_requests) == 1
                assert 'create_quantum_program_and_job' in fake_client.all_stream_requests[0]
        duet.run(test)

    @mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
    def test_submit_expects_job_response(self, client_constructor):
        expected_job = quantum.QuantumJob(name='projects/proj/programs/prog/jobs/job0')
        fake_client, manager = setup(client_constructor)

        async def test():
            async with duet.timeout_scope(5):
                actual_job_future = manager.submit(REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB0)
                await fake_client.wait_for_requests()
                await fake_client.reply(quantum.QuantumRunStreamResponse(job=expected_job))
                actual_job = await actual_job_future
                manager.stop()
                assert actual_job == expected_job
                assert len(fake_client.all_stream_requests) == 1
                assert 'create_quantum_program_and_job' in fake_client.all_stream_requests[0]
        duet.run(test)

    @mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
    def test_submit_job_does_not_exist_expects_create_quantum_job_request(self, client_constructor):
        expected_result = quantum.QuantumResult(parent='projects/proj/programs/prog/jobs/job0')
        fake_client, manager = setup(client_constructor)

        async def test():
            async with duet.timeout_scope(5):
                actual_result_future = manager.submit(REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB0)
                await fake_client.wait_for_requests()
                await fake_client.reply(google_exceptions.ServiceUnavailable('unavailable'))
                await fake_client.wait_for_requests()
                await fake_client.reply(quantum.QuantumRunStreamResponse(error=quantum.StreamError(code=quantum.StreamError.Code.JOB_DOES_NOT_EXIST)))
                await fake_client.wait_for_requests()
                await fake_client.reply(quantum.QuantumRunStreamResponse(result=expected_result))
                actual_result = await actual_result_future
                manager.stop()
                assert actual_result == expected_result
                assert len(fake_client.all_stream_requests) == 3
                assert 'create_quantum_program_and_job' in fake_client.all_stream_requests[0]
                assert 'get_quantum_result' in fake_client.all_stream_requests[1]
                assert 'create_quantum_job' in fake_client.all_stream_requests[2]
        duet.run(test)

    @mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
    def test_submit_program_does_not_exist_expects_create_quantum_program_and_job_request(self, client_constructor):
        expected_result = quantum.QuantumResult(parent='projects/proj/programs/prog/jobs/job0')
        fake_client, manager = setup(client_constructor)

        async def test():
            async with duet.timeout_scope(5):
                actual_result_future = manager.submit(REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB0)
                await fake_client.wait_for_requests()
                await fake_client.reply(google_exceptions.ServiceUnavailable('unavailable'))
                await fake_client.wait_for_requests()
                await fake_client.reply(quantum.QuantumRunStreamResponse(error=quantum.StreamError(code=quantum.StreamError.Code.JOB_DOES_NOT_EXIST)))
                await fake_client.wait_for_requests()
                await fake_client.reply(quantum.QuantumRunStreamResponse(error=quantum.StreamError(code=quantum.StreamError.Code.PROGRAM_DOES_NOT_EXIST)))
                await fake_client.wait_for_requests()
                await fake_client.reply(quantum.QuantumRunStreamResponse(result=expected_result))
                actual_result = await actual_result_future
                manager.stop()
                assert actual_result == expected_result
                assert len(fake_client.all_stream_requests) == 4
                assert 'create_quantum_program_and_job' in fake_client.all_stream_requests[0]
                assert 'get_quantum_result' in fake_client.all_stream_requests[1]
                assert 'create_quantum_job' in fake_client.all_stream_requests[2]
                assert 'create_quantum_program_and_job' in fake_client.all_stream_requests[3]
        duet.run(test)

    @mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
    def test_submit_program_already_exists_expects_get_result_request(self, client_constructor):
        expected_result = quantum.QuantumResult(parent='projects/proj/programs/prog/jobs/job0')
        fake_client, manager = setup(client_constructor)

        async def test():
            async with duet.timeout_scope(5):
                actual_result_future = manager.submit(REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB0)
                await fake_client.wait_for_requests()
                await fake_client.reply(quantum.QuantumRunStreamResponse(error=quantum.StreamError(code=quantum.StreamError.Code.PROGRAM_ALREADY_EXISTS)))
                await fake_client.wait_for_requests()
                await fake_client.reply(quantum.QuantumRunStreamResponse(result=expected_result))
                actual_result = await actual_result_future
                manager.stop()
                assert actual_result == expected_result
                assert len(fake_client.all_stream_requests) == 2
                assert 'create_quantum_program_and_job' in fake_client.all_stream_requests[0]
                assert 'get_quantum_result' in fake_client.all_stream_requests[1]
        duet.run(test)

    @mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
    def test_submit_program_already_exists_but_job_does_not_exist_expects_create_job_request(self, client_constructor):
        expected_result = quantum.QuantumResult(parent='projects/proj/programs/prog/jobs/job0')
        fake_client, manager = setup(client_constructor)

        async def test():
            async with duet.timeout_scope(5):
                actual_result_future = manager.submit(REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB0)
                await fake_client.wait_for_requests()
                await fake_client.reply(quantum.QuantumRunStreamResponse(error=quantum.StreamError(code=quantum.StreamError.Code.PROGRAM_ALREADY_EXISTS)))
                await fake_client.wait_for_requests()
                await fake_client.reply(quantum.QuantumRunStreamResponse(error=quantum.StreamError(code=quantum.StreamError.Code.JOB_DOES_NOT_EXIST)))
                await fake_client.wait_for_requests()
                await fake_client.reply(quantum.QuantumRunStreamResponse(result=expected_result))
                actual_result = await actual_result_future
                manager.stop()
                assert actual_result == expected_result
                assert len(fake_client.all_stream_requests) == 3
                assert 'create_quantum_program_and_job' in fake_client.all_stream_requests[0]
                assert 'get_quantum_result' in fake_client.all_stream_requests[1]
                assert 'create_quantum_job' in fake_client.all_stream_requests[2]
        duet.run(test)

    @mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
    def test_submit_job_already_exist_expects_get_result_request(self, client_constructor):
        """Verifies the behavior when the client receives a JOB_ALREADY_EXISTS error.

        This error is only expected to be triggered in the following race condition:
        1. The client sends a CreateQuantumProgramAndJobRequest.
        2. The client's stream disconnects.
        3. The client retries with a new stream and a GetQuantumResultRequest.
        4. The job doesn't exist yet, and the client receives a "job not found" error.
        5. Scheduler creates the program and job.
        6. The client retries with a CreateJobRequest and fails with a "job already exists" error.

        The JOB_ALREADY_EXISTS error from `CreateQuantumJobRequest` is only possible if the job
        doesn't exist yet at the last `GetQuantumResultRequest`.
        """
        expected_result = quantum.QuantumResult(parent='projects/proj/programs/prog/jobs/job0')
        fake_client, manager = setup(client_constructor)

        async def test():
            async with duet.timeout_scope(5):
                actual_result_future = manager.submit(REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB0)
                await fake_client.wait_for_requests()
                await fake_client.reply(google_exceptions.ServiceUnavailable('unavailable'))
                await fake_client.wait_for_requests()
                await fake_client.reply(quantum.QuantumRunStreamResponse(error=quantum.StreamError(code=quantum.StreamError.Code.JOB_DOES_NOT_EXIST)))
                await fake_client.wait_for_requests()
                await fake_client.reply(quantum.QuantumRunStreamResponse(error=quantum.StreamError(code=quantum.StreamError.Code.JOB_ALREADY_EXISTS)))
                await fake_client.wait_for_requests()
                await fake_client.reply(quantum.QuantumRunStreamResponse(result=expected_result))
                actual_result = await actual_result_future
                manager.stop()
                assert actual_result == expected_result
                assert len(fake_client.all_stream_requests) == 4
                assert 'create_quantum_program_and_job' in fake_client.all_stream_requests[0]
                assert 'get_quantum_result' in fake_client.all_stream_requests[1]
                assert 'create_quantum_job' in fake_client.all_stream_requests[2]
                assert 'get_quantum_result' in fake_client.all_stream_requests[3]
        duet.run(test)

    @mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
    def test_submit_twice_in_parallel_expect_result_responses(self, client_constructor):
        expected_result0 = quantum.QuantumResult(parent='projects/proj/programs/prog/jobs/job0')
        expected_result1 = quantum.QuantumResult(parent='projects/proj/programs/prog/jobs/job1')
        fake_client, manager = setup(client_constructor)

        async def test():
            async with duet.timeout_scope(5):
                actual_result0_future = manager.submit(REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB0)
                actual_result1_future = manager.submit(REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB1)
                await fake_client.wait_for_requests(num_requests=2)
                await fake_client.reply(quantum.QuantumRunStreamResponse(message_id=fake_client.all_stream_requests[0].message_id, result=expected_result0))
                await fake_client.reply(quantum.QuantumRunStreamResponse(message_id=fake_client.all_stream_requests[1].message_id, result=expected_result1))
                actual_result1 = await actual_result1_future
                actual_result0 = await actual_result0_future
                manager.stop()
                assert actual_result0 == expected_result0
                assert actual_result1 == expected_result1
                assert len(fake_client.all_stream_requests) == 2
                assert 'create_quantum_program_and_job' in fake_client.all_stream_requests[0]
                assert 'create_quantum_program_and_job' in fake_client.all_stream_requests[1]
        duet.run(test)

    @mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
    def test_submit_twice_and_break_stream_expect_result_responses(self, client_constructor):
        expected_result0 = quantum.QuantumResult(parent='projects/proj/programs/prog/jobs/job0')
        expected_result1 = quantum.QuantumResult(parent='projects/proj/programs/prog/jobs/job1')
        fake_client, manager = setup(client_constructor)

        async def test():
            async with duet.timeout_scope(5):
                actual_result0_future = manager.submit(REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB0)
                actual_result1_future = manager.submit(REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB1)
                await fake_client.wait_for_requests(num_requests=2)
                await fake_client.reply(google_exceptions.ServiceUnavailable('unavailable'))
                await fake_client.wait_for_requests(num_requests=2)
                await fake_client.reply(quantum.QuantumRunStreamResponse(message_id=next((req.message_id for req in fake_client.all_stream_requests[2:] if req.get_quantum_result.parent == expected_result0.parent)), result=expected_result0))
                await fake_client.reply(quantum.QuantumRunStreamResponse(message_id=next((req.message_id for req in fake_client.all_stream_requests[2:] if req.get_quantum_result.parent == expected_result1.parent)), result=expected_result1))
                actual_result0 = await actual_result0_future
                actual_result1 = await actual_result1_future
                manager.stop()
                assert actual_result0 == expected_result0
                assert actual_result1 == expected_result1
                assert len(fake_client.all_stream_requests) == 4
                assert 'create_quantum_program_and_job' in fake_client.all_stream_requests[0]
                assert 'create_quantum_program_and_job' in fake_client.all_stream_requests[1]
                assert 'get_quantum_result' in fake_client.all_stream_requests[2]
                assert 'get_quantum_result' in fake_client.all_stream_requests[3]
        duet.run(test)

    @mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
    def test_stop_cancels_existing_sends(self, client_constructor):
        fake_client, manager = setup(client_constructor)

        async def test():
            async with duet.timeout_scope(5):
                actual_result_future = manager.submit(REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB0)
                await fake_client.wait_for_requests()
                manager.stop()
                with pytest.raises(concurrent.futures.CancelledError):
                    await actual_result_future
        duet.run(test)

    @mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
    def test_stop_then_send_expects_result_response(self, client_constructor):
        """New requests should work after stopping the manager."""
        expected_result = quantum.QuantumResult(parent='projects/proj/programs/prog/jobs/job0')
        fake_client, manager = setup(client_constructor)

        async def test():
            async with duet.timeout_scope(5):
                manager.stop()
                actual_result_future = manager.submit(REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB0)
                await fake_client.wait_for_requests()
                await fake_client.reply(quantum.QuantumRunStreamResponse(result=expected_result))
                actual_result = await actual_result_future
                manager.stop()
                assert actual_result == expected_result
                assert len(fake_client.all_stream_requests) == 1
                assert 'create_quantum_program_and_job' in fake_client.all_stream_requests[0]
        duet.run(test)

    @pytest.mark.parametrize('error_code, current_request_type', [(Code.PROGRAM_DOES_NOT_EXIST, 'create_quantum_program_and_job'), (Code.PROGRAM_DOES_NOT_EXIST, 'get_quantum_result'), (Code.PROGRAM_ALREADY_EXISTS, 'create_quantum_job'), (Code.PROGRAM_ALREADY_EXISTS, 'get_quantum_result'), (Code.JOB_DOES_NOT_EXIST, 'create_quantum_program_and_job'), (Code.JOB_DOES_NOT_EXIST, 'create_quantum_job'), (Code.JOB_ALREADY_EXISTS, 'get_quantum_result')])
    def test_get_retry_request_or_raise_expects_stream_error(self, error_code, current_request_type):
        create_quantum_program_and_job_request = quantum.QuantumRunStreamRequest(create_quantum_program_and_job=quantum.CreateQuantumProgramAndJobRequest())
        create_quantum_job_request = quantum.QuantumRunStreamRequest(create_quantum_job=quantum.CreateQuantumJobRequest())
        get_quantum_result_request = quantum.QuantumRunStreamRequest(get_quantum_result=quantum.GetQuantumResultRequest())
        if current_request_type == 'create_quantum_program_and_job':
            current_request = create_quantum_program_and_job_request
        elif current_request_type == 'create_quantum_job':
            current_request = create_quantum_job_request
        elif current_request_type == 'get_quantum_result':
            current_request = get_quantum_result_request
        with pytest.raises(StreamError):
            _get_retry_request_or_raise(quantum.StreamError(code=error_code), current_request, create_quantum_program_and_job_request, create_quantum_job_request, get_quantum_result_request)

    @mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
    def test_broken_stream_stops_request_iterator(self, client_constructor):
        expected_result = quantum.QuantumResult(parent='projects/proj/programs/prog/jobs/job0')
        fake_client, manager = setup(client_constructor)

        async def test():
            async with duet.timeout_scope(5):
                actual_result_future = manager.submit(REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB0)
                await fake_client.wait_for_requests()
                await fake_client.reply(quantum.QuantumRunStreamResponse(message_id=fake_client.all_stream_requests[0].message_id, result=expected_result))
                await actual_result_future
                await fake_client.reply(google_exceptions.ServiceUnavailable('service unavailable'))
                await fake_client.wait_for_request_iterator_stop()
                manager.stop()
        duet.run(test)

    @mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
    def test_stop_stops_request_iterator(self, client_constructor):
        expected_result = quantum.QuantumResult(parent='projects/proj/programs/prog/jobs/job0')
        fake_client, manager = setup(client_constructor)

        async def test():
            async with duet.timeout_scope(5):
                actual_result_future = manager.submit(REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB0)
                await fake_client.wait_for_requests()
                await fake_client.reply(quantum.QuantumRunStreamResponse(message_id=fake_client.all_stream_requests[0].message_id, result=expected_result))
                await actual_result_future
                manager.stop()
                await fake_client.wait_for_request_iterator_stop()
        duet.run(test)

    @mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
    def test_submit_after_stream_breakage(self, client_constructor):
        expected_result0 = quantum.QuantumResult(parent='projects/proj/programs/prog/jobs/job0')
        expected_result1 = quantum.QuantumResult(parent='projects/proj/programs/prog/jobs/job1')
        fake_client, manager = setup(client_constructor)

        async def test():
            async with duet.timeout_scope(5):
                actual_result0_future = manager.submit(REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB0)
                await fake_client.wait_for_requests()
                await fake_client.reply(quantum.QuantumRunStreamResponse(message_id=fake_client.all_stream_requests[0].message_id, result=expected_result0))
                actual_result0 = await actual_result0_future
                await fake_client.reply(google_exceptions.ServiceUnavailable('service unavailable'))
                actual_result1_future = manager.submit(REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB0)
                await fake_client.wait_for_requests()
                await fake_client.reply(quantum.QuantumRunStreamResponse(message_id=fake_client.all_stream_requests[1].message_id, result=expected_result1))
                actual_result1 = await actual_result1_future
                manager.stop()
                assert len(fake_client.all_stream_requests) == 2
                assert 'create_quantum_program_and_job' in fake_client.all_stream_requests[0]
                assert 'create_quantum_program_and_job' in fake_client.all_stream_requests[1]
                assert actual_result0 == expected_result0
                assert actual_result1 == expected_result1
        duet.run(test)