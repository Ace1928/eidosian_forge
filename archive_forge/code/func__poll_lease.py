import logging
from oslo_concurrency import lockutils
from oslo_context import context
from oslo_utils import excutils
from oslo_utils import reflection
from oslo_vmware._i18n import _
from oslo_vmware.common import loopingcall
from oslo_vmware import exceptions
from oslo_vmware import pbm
from oslo_vmware import vim
from oslo_vmware import vim_util
def _poll_lease(self, lease):
    """Poll the state of the given lease.

        When the lease is ready, the event (param done) is notified. In case
        of any error, appropriate exception is set in the event.

        :param lease: lease whose state is to be polled
        """
    try:
        state = self.invoke_api(vim_util, 'get_object_property', self.vim, lease, 'state', skip_op_id=True)
    except exceptions.VimException:
        with excutils.save_and_reraise_exception():
            LOG.exception('Error occurred while checking state of lease: %s.', lease)
    else:
        if state == 'ready':
            LOG.debug('Lease: %s is ready.', lease)
            raise loopingcall.LoopingCallDone()
        elif state == 'initializing':
            LOG.debug('Lease: %s is initializing.', lease)
        elif state == 'error':
            LOG.debug('Invoking VIM API to read lease: %s error.', lease)
            error_msg = self._get_error_message(lease)
            excep_msg = _('Lease: %(lease)s is in error state. Details: %(error_msg)s.') % {'lease': lease, 'error_msg': error_msg}
            LOG.error(excep_msg)
            raise exceptions.translate_fault(error_msg, excep_msg)
        else:
            excep_msg = _('Unknown state: %(state)s for lease: %(lease)s.') % {'state': state, 'lease': lease}
            LOG.error(excep_msg)
            raise exceptions.VimException(excep_msg)