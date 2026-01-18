import logging
from osprofiler.drivers import base
def _noop_notifier(info, context=None):
    """Do nothing on notify()."""