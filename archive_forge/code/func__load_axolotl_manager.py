from yowsup.config.manager import ConfigManager
from yowsup.config.v1.config import Config
from yowsup.axolotl.manager import AxolotlManager
from yowsup.axolotl.factory import AxolotlManagerFactory
import logging
def _load_axolotl_manager(self):
    return AxolotlManagerFactory().get_manager(self._profile_name, self.username)