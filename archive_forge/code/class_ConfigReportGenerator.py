from oslo_config import cfg
from oslo_reports.models import conf as cm
class ConfigReportGenerator(object):
    """A Configuration Data Generator

    This generator returns
    :class:`oslo_reports.models.conf.ConfigModel`,
    by default using the configuration options stored
    in :attr:`oslo_config.cfg.CONF`, which is where
    OpenStack stores everything.

    :param cnf: the configuration option object
    :type cnf: :class:`oslo_config.cfg.ConfigOpts`
    """

    def __init__(self, cnf=cfg.CONF):
        self.conf_obj = cnf

    def __call__(self):
        return cm.ConfigModel(self.conf_obj)