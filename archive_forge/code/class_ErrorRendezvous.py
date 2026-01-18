from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import contextlib
import sys
import threading
import time
import six
import tensorflow as tf
from tensorflow_estimator.python.estimator.tools import analytics
class ErrorRendezvous(object):
    """Resolve errors from multiple threads during TPU execution.

  TPU errors can occur on the infeed or outfeed threads as well as the main
  training thread.

  Depending on which thread "wins" and receives the session error first, we may
  end up showing users a confusing and non-actionable error message (session
  cancelled) instead of a root cause (e.g. a bad filename).

  The rendezvous object provides a location to capture these errors until all
  threads terminate.  At that point we can choose the most informative error
  to report.
  """

    def __init__(self, num_sources):
        self._errors = {}
        self._num_sources = num_sources
        self._session_cancel_timer = None

    def record_error(self, source, exc_info, session=None):
        """Report an exception from the given source.

    If a session is passed, a timer will be registered to close it after a few
    seconds.  This is necessary to ensure the main training loop does not hang
    if an infeed/oufeed error occurs.  We sleep a few seconds to allow a more
    interesting error from another thread to propagate.

    Args:
      source: string, source of the error
      exc_info: Output from `sys.exc_info` (type, value, traceback)
      session: Session to close after delay.
    """
        _, value, _ = exc_info
        if isinstance(value, _IGNORED_ERRORS):
            return
        self._errors[source] = exc_info
        try:
            if value.op.type == _CHECK_NUMERIC_OP_NAME:
                analytics.track_numerical_issues(exc_info)
                return
        except AttributeError as _:
            pass
        if session is not None and self._session_cancel_timer is None:

            def _cancel_session():
                time.sleep(5)
                tf.compat.v1.logging.error('Closing session due to error %s' % value)
                try:
                    session.close()
                except:
                    tf.compat.v1.logging.error('\n\n\nFailed to close session after error.Other threads may hang.\n\n\n')
            self._session_cancel_timer = threading.Thread(target=_cancel_session)
            self._session_cancel_timer.daemon = True
            self._session_cancel_timer.start()

    def record_done(self, source):
        """Mark execution source `source` as done.

    If an error was originally reported from `source` it is left intact.

    Args:
      source: `str`, source being recorded
    """
        tf.compat.v1.logging.info('%s marked as finished', source)
        if source not in self._errors:
            self._errors[source] = None

    @contextlib.contextmanager
    def catch_errors(self, source, session=None):
        """Context manager to report any errors within a block."""
        try:
            yield
        except Exception:
            self.record_error(source, sys.exc_info(), session)

    def raise_errors(self, timeout_sec=0):
        """Wait for up to `timeout` seconds for all error sources to finish.

    Preferentially raise "interesting" errors (errors not in the
    _UNINTERESTING_ERRORS) set.

    Args:
      timeout_sec: Seconds to wait for other error sources.
    """
        for _ in range(timeout_sec):
            if len(self._errors) == self._num_sources:
                break
            time.sleep(1)
        kept_errors = [(k, v) for k, v in self._errors.items() if v is not None]
        for k, (typ, value, traceback) in kept_errors:
            if isinstance(value, _UNINTERESTING_ERRORS):
                continue
            else:
                tf.compat.v1.logging.warn('Reraising captured error')
                six.reraise(typ, value, traceback)
        for k, (typ, value, traceback) in kept_errors:
            tf.compat.v1.logging.warn('Reraising captured error')
            six.reraise(typ, value, traceback)