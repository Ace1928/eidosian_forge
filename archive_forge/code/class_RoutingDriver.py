import fnmatch
import logging
from oslo_config import cfg
from stevedore import dispatch
import yaml
from oslo_messaging.notify import notifier
class RoutingDriver(notifier.Driver):
    NOTIFIER_PLUGIN_NAMESPACE = 'oslo.messaging.notify.drivers'
    plugin_manager = None
    routing_groups = None
    used_drivers = None

    def _should_load_plugin(self, ext, *args, **kwargs):
        if ext.name == 'routing':
            return False
        return ext.name in self.used_drivers

    def _get_notifier_config_file(self, filename):
        """Broken out for testing."""
        return open(filename, 'r')

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

    def _get_drivers_for_message(self, group, event_type, priority):
        """Which drivers should be called for this event_type
           or priority.
        """
        accepted_drivers = set()
        for driver, rules in group.items():
            checks = []
            for key, patterns in rules.items():
                if key == 'accepted_events':
                    c = [fnmatch.fnmatch(event_type, p) for p in patterns]
                    checks.append(any(c))
                if key == 'accepted_priorities':
                    c = [fnmatch.fnmatch(priority, p.lower()) for p in patterns]
                    checks.append(any(c))
            if all(checks):
                accepted_drivers.add(driver)
        return list(accepted_drivers)

    def _filter_func(self, ext, context, message, priority, retry, accepted_drivers):
        """True/False if the driver should be called for this message.
        """
        return ext.name in accepted_drivers

    def _call_notify(self, ext, context, message, priority, retry, accepted_drivers):
        """Emit the notification.
        """
        LOG.info("Routing '%(event)s' notification to '%(driver)s' driver", {'event': message.get('event_type'), 'driver': ext.name})
        ext.obj.notify(context, message, priority, retry)

    def notify(self, context, message, priority, retry):
        if not self.plugin_manager:
            self._load_notifiers()
        event_type = message['event_type']
        accepted_drivers = set()
        for group in self.routing_groups.values():
            accepted_drivers.update(self._get_drivers_for_message(group, event_type, priority.lower()))
        self.plugin_manager.map(self._filter_func, self._call_notify, context, message, priority, retry, list(accepted_drivers))