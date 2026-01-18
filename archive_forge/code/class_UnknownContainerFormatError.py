import re
from io import BytesIO
from .. import errors
class UnknownContainerFormatError(ContainerError):
    _fmt = 'Unrecognised container format: %(container_format)r'

    def __init__(self, container_format):
        self.container_format = container_format