from .. import errors
from ..utils.utils import (
from .base import DictType
from .healthcheck import Healthcheck
class LogConfigTypesEnum:
    _values = ('json-file', 'syslog', 'journald', 'gelf', 'fluentd', 'none')
    JSON, SYSLOG, JOURNALD, GELF, FLUENTD, NONE = _values