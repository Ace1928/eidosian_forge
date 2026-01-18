import abc
import logging
import threading
import time
from contextlib import contextmanager
from inspect import getframeinfo, stack
from typing import Any, Dict, List, Optional, Set
class TimerServer(abc.ABC):
    """
    Entity that monitors active timers and expires them
    in a timely fashion. This server is responsible for
    reaping workers that have expired timers.
    """

    def __init__(self, request_queue: RequestQueue, max_interval: float, daemon: bool=True):
        """
        :param request_queue: Consumer ``RequestQueue``
        :param max_interval: max time (in seconds) to wait
                             for an item in the request_queue
        :param daemon: whether to run the watchdog thread as a daemon
        """
        super().__init__()
        self._request_queue = request_queue
        self._max_interval = max_interval
        self._daemon = daemon
        self._watchdog_thread: Optional[threading.Thread] = None
        self._stop_signaled = False

    @abc.abstractmethod
    def register_timers(self, timer_requests: List[TimerRequest]) -> None:
        """
        Processes the incoming timer requests and registers them with the server.
        The timer request can either be a acquire-timer or release-timer request.
        Timer requests with a negative expiration_time should be interpreted
        as a release-timer request.
        """
        pass

    @abc.abstractmethod
    def clear_timers(self, worker_ids: Set[Any]) -> None:
        """
        Clears all timers for the given ``worker_ids``.
        """
        pass

    @abc.abstractmethod
    def get_expired_timers(self, deadline: float) -> Dict[str, List[TimerRequest]]:
        """
        Returns all expired timers for each worker_id. An expired timer
        is a timer for which the expiration_time is less than or equal to
        the provided deadline.
        """
        pass

    @abc.abstractmethod
    def _reap_worker(self, worker_id: Any) -> bool:
        """
        Reaps the given worker. Returns True if the worker has been
        successfully reaped, False otherwise. If any uncaught exception
        is thrown from this method, the worker is considered reaped
        and all associated timers will be removed.
        """

    def _reap_worker_no_throw(self, worker_id: Any) -> bool:
        """
        Wraps ``_reap_worker(worker_id)``, if an uncaught exception is
        thrown, then it considers the worker as reaped.
        """
        try:
            return self._reap_worker(worker_id)
        except Exception:
            log.exception('Uncaught exception thrown from _reap_worker(), check that the implementation correctly catches exceptions')
            return True

    def _watchdog_loop(self):
        while not self._stop_signaled:
            try:
                self._run_watchdog()
            except Exception:
                log.exception('Error running watchdog')

    def _run_watchdog(self):
        batch_size = max(1, self._request_queue.size())
        timer_requests = self._request_queue.get(batch_size, self._max_interval)
        self.register_timers(timer_requests)
        now = time.time()
        reaped_worker_ids = set()
        for worker_id, expired_timers in self.get_expired_timers(now).items():
            log.info('Reaping worker_id=[%s]. Expired timers: %s', worker_id, self._get_scopes(expired_timers))
            if self._reap_worker_no_throw(worker_id):
                log.info('Successfully reaped worker=[%s]', worker_id)
                reaped_worker_ids.add(worker_id)
            else:
                log.error('Error reaping worker=[%s]. Will retry on next watchdog.', worker_id)
        self.clear_timers(reaped_worker_ids)

    def _get_scopes(self, timer_requests):
        return [r.scope_id for r in timer_requests]

    def start(self) -> None:
        log.info('Starting %s... max_interval=%s, daemon=%s', type(self).__name__, self._max_interval, self._daemon)
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=self._daemon)
        log.info('Starting watchdog thread...')
        self._watchdog_thread.start()

    def stop(self) -> None:
        log.info('Stopping %s', type(self).__name__)
        self._stop_signaled = True
        if self._watchdog_thread:
            log.info('Stopping watchdog thread...')
            self._watchdog_thread.join(self._max_interval)
            self._watchdog_thread = None
        else:
            log.info('No watchdog thread running, doing nothing')