import datetime
from heat.common import exception
from heat.common.i18n import _
from heat.engine import resource
from oslo_log import log as logging
from oslo_utils import timeutils
class CooldownMixin(object):
    """Utility class to encapsulate Cooldown related logic.

    This logic includes both cooldown timestamp comparing and scaling in
    progress checking.
    """

    def _sanitize_cooldown(self, cooldown):
        if cooldown is None:
            return 0
        return max(0, cooldown)

    def _check_scaling_allowed(self, cooldown):
        metadata = self.metadata_get()
        if metadata.get('scaling_in_progress'):
            LOG.info('Can not perform scaling action: resource %s is already in scaling.', self.name)
            reason = _('due to scaling activity')
            raise resource.NoActionRequired(res_name=self.name, reason=reason)
        cooldown = self._sanitize_cooldown(cooldown)
        if all((k not in metadata for k in ('cooldown', 'cooldown_end'))):
            metadata.pop('scaling_in_progress', None)
            if metadata and cooldown != 0:
                last_adjust = next(iter(metadata.keys()))
                if not timeutils.is_older_than(last_adjust, cooldown):
                    self._log_and_raise_no_action(cooldown)
        elif 'cooldown_end' in metadata:
            cooldown_end = next(iter(metadata['cooldown_end'].keys()))
            now = timeutils.utcnow().isoformat()
            if now < cooldown_end:
                self._log_and_raise_no_action(cooldown)
        elif cooldown != 0:
            last_adjust = next(iter(metadata['cooldown'].keys()))
            if not timeutils.is_older_than(last_adjust, cooldown):
                self._log_and_raise_no_action(cooldown)
        metadata['scaling_in_progress'] = True
        self.metadata_set(metadata)

    def _log_and_raise_no_action(self, cooldown):
        LOG.info('Can not perform scaling action: resource %(name)s is in cooldown (%(cooldown)s).', {'name': self.name, 'cooldown': cooldown})
        reason = _('due to cooldown, cooldown %s') % cooldown
        raise resource.NoActionRequired(res_name=self.name, reason=reason)

    def _finished_scaling(self, cooldown, cooldown_reason, size_changed=True):
        metadata = self.metadata_get()
        if size_changed:
            cooldown = self._sanitize_cooldown(cooldown)
            cooldown_end = (timeutils.utcnow() + datetime.timedelta(seconds=cooldown)).isoformat()
            if 'cooldown_end' in metadata:
                cooldown_end = max(next(iter(metadata['cooldown_end'].keys())), cooldown_end)
            metadata['cooldown_end'] = {cooldown_end: cooldown_reason}
        metadata['scaling_in_progress'] = False
        try:
            self.metadata_set(metadata)
        except exception.NotFound:
            pass

    def handle_metadata_reset(self):
        metadata = self.metadata_get()
        if 'scaling_in_progress' in metadata:
            metadata['scaling_in_progress'] = False
            self.metadata_set(metadata)