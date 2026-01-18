import time
from boto.compat import json
class IndexFieldStatus(OptionStatus):

    def _update_options(self, options):
        self.update(options)

    def save(self):
        pass