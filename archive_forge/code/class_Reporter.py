import logging
class Reporter:
    _instance = None

    def __init__(self, settings=None):
        if Reporter._instance is not None:
            return
        if settings is None:
            logging.error('internal issue: reporter not setup')
        Reporter._instance = _Reporter(settings)

    def __getattr__(self, name):
        return getattr(self._instance, name)