from __future__ import absolute_import
import weakref
from sentry_sdk.consts import OP
from sentry_sdk.api import continue_trace
from sentry_sdk.hub import Hub
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk.integrations.logging import ignore_logger
from sentry_sdk.tracing import TRANSACTION_SOURCE_TASK
from sentry_sdk.utils import (
from sentry_sdk._types import TYPE_CHECKING
class RqIntegration(Integration):
    identifier = 'rq'

    @staticmethod
    def setup_once():
        version = parse_version(RQ_VERSION)
        if version is None:
            raise DidNotEnable('Unparsable RQ version: {}'.format(RQ_VERSION))
        if version < (0, 6):
            raise DidNotEnable('RQ 0.6 or newer is required.')
        old_perform_job = Worker.perform_job

        def sentry_patched_perform_job(self, job, *args, **kwargs):
            hub = Hub.current
            integration = hub.get_integration(RqIntegration)
            if integration is None:
                return old_perform_job(self, job, *args, **kwargs)
            client = hub.client
            assert client is not None
            with hub.push_scope() as scope:
                scope.clear_breadcrumbs()
                scope.add_event_processor(_make_event_processor(weakref.ref(job)))
                transaction = continue_trace(job.meta.get('_sentry_trace_headers') or {}, op=OP.QUEUE_TASK_RQ, name='unknown RQ task', source=TRANSACTION_SOURCE_TASK)
                with capture_internal_exceptions():
                    transaction.name = job.func_name
                with hub.start_transaction(transaction, custom_sampling_context={'rq_job': job}):
                    rv = old_perform_job(self, job, *args, **kwargs)
            if self.is_horse:
                client.flush()
            return rv
        Worker.perform_job = sentry_patched_perform_job
        old_handle_exception = Worker.handle_exception

        def sentry_patched_handle_exception(self, job, *exc_info, **kwargs):
            if job._status == JobStatus.FAILED or job.is_failed:
                _capture_exception(exc_info)
            return old_handle_exception(self, job, *exc_info, **kwargs)
        Worker.handle_exception = sentry_patched_handle_exception
        old_enqueue_job = Queue.enqueue_job

        def sentry_patched_enqueue_job(self, job, **kwargs):
            hub = Hub.current
            if hub.get_integration(RqIntegration) is not None:
                if hub.scope.span is not None:
                    job.meta['_sentry_trace_headers'] = dict(hub.iter_trace_propagation_headers())
            return old_enqueue_job(self, job, **kwargs)
        Queue.enqueue_job = sentry_patched_enqueue_job
        ignore_logger('rq.worker')