import re
from io import BytesIO
from .. import errors
class ContainerHasExcessDataError(ContainerError):
    _fmt = 'Container has data after end marker: %(excess)r'

    def __init__(self, excess):
        self.excess = excess