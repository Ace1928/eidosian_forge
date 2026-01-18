from aniso8601 import compat
from aniso8601.builders import TupleBuilder
from aniso8601.builders.python import PythonTimeBuilder
from aniso8601.date import parse_date
from aniso8601.decimalfraction import normalize
from aniso8601.exceptions import ISOFormatError
from aniso8601.resolution import DurationResolution
from aniso8601.time import parse_time
def _parse_duration_prescribed_notime(isodurationstr):
    durationstr = normalize(isodurationstr)
    yearstr = None
    monthstr = None
    daystr = None
    weekstr = None
    weekidx = durationstr.find('W')
    yearidx = durationstr.find('Y')
    monthidx = durationstr.find('M')
    dayidx = durationstr.find('D')
    if weekidx != -1:
        weekstr = durationstr[1:-1]
    elif yearidx != -1 and monthidx != -1 and (dayidx != -1):
        yearstr = durationstr[1:yearidx]
        monthstr = durationstr[yearidx + 1:monthidx]
        daystr = durationstr[monthidx + 1:-1]
    elif yearidx != -1 and monthidx != -1:
        yearstr = durationstr[1:yearidx]
        monthstr = durationstr[yearidx + 1:monthidx]
    elif yearidx != -1 and dayidx != -1:
        yearstr = durationstr[1:yearidx]
        daystr = durationstr[yearidx + 1:dayidx]
    elif monthidx != -1 and dayidx != -1:
        monthstr = durationstr[1:monthidx]
        daystr = durationstr[monthidx + 1:-1]
    elif yearidx != -1:
        yearstr = durationstr[1:-1]
    elif monthidx != -1:
        monthstr = durationstr[1:-1]
    elif dayidx != -1:
        daystr = durationstr[1:-1]
    else:
        raise ISOFormatError('"{0}" is not a valid ISO 8601 duration.'.format(isodurationstr))
    for componentstr in [yearstr, monthstr, daystr, weekstr]:
        if componentstr is not None:
            if '.' in componentstr:
                intstr, fractionalstr = componentstr.split('.', 1)
                if intstr.isdigit() is False:
                    raise ISOFormatError('"{0}" is not a valid ISO 8601 duration.'.format(isodurationstr))
            elif componentstr.isdigit() is False:
                raise ISOFormatError('"{0}" is not a valid ISO 8601 duration.'.format(isodurationstr))
    return {'PnY': yearstr, 'PnM': monthstr, 'PnW': weekstr, 'PnD': daystr}