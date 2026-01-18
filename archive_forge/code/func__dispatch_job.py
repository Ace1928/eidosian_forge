import abc
import contextlib
import functools
import itertools
import threading
from oslo_utils import excutils
from oslo_utils import timeutils
from taskflow.conductors import base
from taskflow import exceptions as excp
from taskflow.listeners import logging as logging_listener
from taskflow import logging
from taskflow import states
from taskflow.types import timing as tt
from taskflow.utils import iter_utils
from taskflow.utils import misc
def _dispatch_job(self, job):
    engine = self._engine_from_job(job)
    listeners = self._listeners_from_job(job, engine)
    with contextlib.ExitStack() as stack:
        for listener in listeners:
            stack.enter_context(listener)
        self._log.debug("Dispatching engine for job '%s'", job)
        consume = True
        details = {'job': job, 'engine': engine, 'conductor': self}

        def _run_engine():
            has_suspended = False
            for _state in engine.run_iter():
                if not has_suspended and self._wait_timeout.is_stopped():
                    self._log.info('Conductor stopped, requesting suspension of engine running job %s', job)
                    engine.suspend()
                    has_suspended = True
        try:
            for stage_func, event_name in [(engine.compile, 'compilation'), (engine.prepare, 'preparation'), (engine.validate, 'validation'), (_run_engine, 'running')]:
                self._notifier.notify('%s_start' % event_name, details)
                stage_func()
                self._notifier.notify('%s_end' % event_name, details)
        except excp.WrappedFailure as e:
            if all((f.check(*self.NO_CONSUME_EXCEPTIONS) for f in e)):
                consume = False
            if self._log.isEnabledFor(logging.WARNING):
                if consume:
                    self._log.warn('Job execution failed (consumption being skipped): %s [%s failures]', job, len(e))
                else:
                    self._log.warn('Job execution failed (consumption proceeding): %s [%s failures]', job, len(e))
                for i, f in enumerate(e):
                    self._log.warn('%s. %s', i + 1, f.pformat(traceback=True))
        except self.NO_CONSUME_EXCEPTIONS:
            self._log.warn('Job execution failed (consumption being skipped): %s', job, exc_info=True)
            consume = False
        except Exception:
            self._log.warn('Job execution failed (consumption proceeding): %s', job, exc_info=True)
        else:
            if engine.storage.get_flow_state() == states.SUSPENDED:
                self._log.info('Job execution was suspended: %s', job)
                consume = False
            else:
                self._log.info('Job completed successfully: %s', job)
        return consume