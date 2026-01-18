import re
from io import BytesIO
from .. import errors
class DuplicateRecordNameError(ContainerError):
    _fmt = 'Container has multiple records with the same name: %(name)s'

    def __init__(self, name):
        self.name = name.decode('utf-8')