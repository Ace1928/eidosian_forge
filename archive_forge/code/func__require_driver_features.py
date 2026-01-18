import logging
from debtcollector import removals
from oslo_config import cfg
from stevedore import driver
from urllib import parse
from oslo_messaging import exceptions
def _require_driver_features(self, requeue=False):
    self._driver.require_features(requeue=requeue)