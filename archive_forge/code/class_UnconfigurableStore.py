from glance_store import driver
from glance_store import exceptions
class UnconfigurableStore(driver.Store):

    def configure(self, re_raise_bsc=False):
        raise exceptions.BadStoreConfiguration()