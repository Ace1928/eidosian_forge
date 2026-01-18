import logging
from string import Template
class DollarMessage(object):

    def __init__(self, fmt, **kwargs):
        self.fmt = fmt
        self.kwargs = kwargs
        self.str = None

    def __str__(self):
        if self.str is None:
            self.str = Template(self.fmt).substitute(**self.kwargs)
        return self.str