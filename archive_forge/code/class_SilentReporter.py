from distutils.core import Command
from distutils.errors import DistutilsSetupError
class SilentReporter(Reporter):

    def __init__(self, source, report_level, halt_level, stream=None, debug=0, encoding='ascii', error_handler='replace'):
        self.messages = []
        Reporter.__init__(self, source, report_level, halt_level, stream, debug, encoding, error_handler)

    def system_message(self, level, message, *children, **kwargs):
        self.messages.append((level, message, children, kwargs))
        return nodes.system_message(message, *children, level=level, type=self.levels[level], **kwargs)