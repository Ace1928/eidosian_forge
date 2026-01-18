import copy
from oslo_config import cfg
from oslo_log import log as logging
import stevedore
from glance.i18n import _, _LE
from the first responsive active location it finds in this list.
def _load_strategies():
    """Load all strategy modules."""
    modules = {}
    namespace = 'glance.common.image_location_strategy.modules'
    ex = stevedore.extension.ExtensionManager(namespace)
    for module_name in ex.names():
        try:
            mgr = stevedore.driver.DriverManager(namespace=namespace, name=module_name, invoke_on_load=False)
            strategy_name = str(mgr.driver.get_strategy_name())
            if strategy_name in modules:
                msg = _('%(strategy)s is registered as a module twice. %(module)s is not being used.') % {'strategy': strategy_name, 'module': module_name}
                LOG.warning(msg)
            else:
                mgr.driver.init()
                modules[strategy_name] = mgr.driver
        except Exception as e:
            LOG.error(_LE('Failed to load location strategy module %(module)s: %(e)s'), {'module': module_name, 'e': e})
    return modules