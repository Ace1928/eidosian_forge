from oslotest import base
from oslo_config import cfg
from oslo_config import fixture as config
def _make_fixture(self):
    conf = cfg.ConfigOpts()
    config_fixture = config.Config(conf)
    config_fixture.setUp()
    config_fixture.register_opt(cfg.StrOpt('testing_option', default='initial_value'))
    config_fixture.register_opt(cfg.IntOpt('test2', min=0, default=5))
    config_fixture.register_opt(cfg.StrOpt('test3', choices=['a', 'b'], default='a'))
    return config_fixture