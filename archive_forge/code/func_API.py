from castellan.key_manager import migration
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import importutils
from stevedore import driver
from stevedore import exception
def API(configuration=None):
    conf = configuration or cfg.CONF
    conf.register_opts(key_manager_opts, group='key_manager')
    try:
        mgr = driver.DriverManager('castellan.drivers', conf.key_manager.backend, invoke_on_load=True, invoke_args=[conf])
        key_mgr = mgr.driver
    except exception.NoMatches:
        LOG.warning('Deprecation Warning : %s is not a stevedore based driver, trying to load it as a class', conf.key_manager.backend)
        cls = importutils.import_class(conf.key_manager.backend)
        key_mgr = cls(configuration=conf)
    return migration.handle_migration(conf, key_mgr)