from aniso8601 import compat
from aniso8601.builders import TupleBuilder
from aniso8601.builders.python import PythonTimeBuilder
from aniso8601.date import parse_date
from aniso8601.decimalfraction import normalize
from aniso8601.exceptions import ISOFormatError
from aniso8601.resolution import DurationResolution
from aniso8601.time import parse_time
def _has_any_component(durationstr, components):
    for component in components:
        if durationstr.find(component) != -1:
            return True
    return False