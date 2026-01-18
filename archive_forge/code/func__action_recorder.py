import collections
import contextlib
import datetime as dt
import itertools
import pydoc
import re
import tenacity
import weakref
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import reflection
from heat.common import exception
from heat.common.i18n import _
from heat.common import identifier
from heat.common import short_id
from heat.common import timeutils
from heat.engine import attributes
from heat.engine.cfn import template as cfn_tmpl
from heat.engine import clients
from heat.engine.clients import default_client_plugin
from heat.engine import environment
from heat.engine import event
from heat.engine import function
from heat.engine.hot import template as hot_tmpl
from heat.engine import node_data
from heat.engine import properties
from heat.engine import resources
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import status
from heat.engine import support
from heat.engine import sync_point
from heat.engine import template
from heat.objects import resource as resource_objects
from heat.objects import resource_data as resource_data_objects
from heat.objects import resource_properties_data as rpd_objects
from heat.rpc import client as rpc_client
@contextlib.contextmanager
def _action_recorder(self, action, expected_exceptions=tuple()):
    """Return a context manager to record the progress of an action.

        Upon entering the context manager, the state is set to IN_PROGRESS.
        Upon exiting, the state will be set to COMPLETE if no exception was
        raised, or FAILED otherwise. Non-exit exceptions will be translated
        to ResourceFailure exceptions.

        Expected exceptions are re-raised, with the Resource moved to the
        COMPLETE state.
        """
    attempts = 1
    first_iter = [True]
    if self.stack.convergence:
        if self._should_lock_on_action(action):
            lock_acquire = self.LOCK_ACQUIRE
            lock_release = self.LOCK_RELEASE
        else:
            lock_acquire = lock_release = self.LOCK_RESPECT
        if action != self.CREATE:
            attempts += max(cfg.CONF.client_retry_limit, 0)
    else:
        lock_acquire = lock_release = self.LOCK_NONE

    @tenacity.retry(stop=tenacity.stop_after_attempt(attempts), retry=tenacity.retry_if_exception_type(exception.UpdateInProgress), wait=tenacity.wait_random(max=2), reraise=True)
    def set_in_progress():
        if not first_iter[0]:
            res_obj = resource_objects.Resource.get_obj(self.context, self.id)
            self._atomic_key = res_obj.atomic_key
        else:
            first_iter[0] = False
        self.state_set(action, self.IN_PROGRESS, lock=lock_acquire)
    try:
        set_in_progress()
        yield
    except exception.UpdateInProgress:
        with excutils.save_and_reraise_exception():
            LOG.info('Update in progress for %s', self.name)
    except expected_exceptions as ex:
        with excutils.save_and_reraise_exception():
            self.state_set(action, self.COMPLETE, str(ex), lock=lock_release)
            LOG.debug('%s', str(ex))
    except Exception as ex:
        LOG.info('%(action)s: %(info)s', {'action': action, 'info': str(self)}, exc_info=True)
        failure = exception.ResourceFailure(ex, self, action)
        self.state_set(action, self.FAILED, str(failure), lock=lock_release)
        raise failure
    except BaseException as exc:
        with excutils.save_and_reraise_exception():
            try:
                reason = str(exc)
                msg = '%s aborted' % action
                if reason:
                    msg += ' (%s)' % reason
                self.state_set(action, self.FAILED, msg, lock=lock_release)
            except Exception:
                LOG.exception('Error marking resource as failed')
    else:
        self.state_set(action, self.COMPLETE, lock=lock_release)