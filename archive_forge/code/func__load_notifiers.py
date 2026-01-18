import fnmatch
import logging
from oslo_config import cfg
from stevedore import dispatch
import yaml
from oslo_messaging.notify import notifier
def _load_notifiers(self):
    """One-time load of notifier config file."""
    self.routing_groups = {}
    self.used_drivers = set()
    filename = CONF.oslo_messaging_notifications.routing_config
    if not filename:
        return
    self.routing_groups = yaml.safe_load(self._get_notifier_config_file(filename))
    if not self.routing_groups:
        self.routing_groups = {}
        return
    for group in self.routing_groups.values():
        self.used_drivers.update(group.keys())
    LOG.debug('loading notifiers from %s', self.NOTIFIER_PLUGIN_NAMESPACE)
    self.plugin_manager = dispatch.DispatchExtensionManager(namespace=self.NOTIFIER_PLUGIN_NAMESPACE, check_func=self._should_load_plugin, invoke_on_load=True, invoke_args=None)
    if not list(self.plugin_manager):
        LOG.warning('Failed to load any notifiers for %s', self.NOTIFIER_PLUGIN_NAMESPACE)