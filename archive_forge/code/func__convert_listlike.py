from __future__ import annotations
from datetime import (
from typing import TYPE_CHECKING
import warnings
import numpy as np
from pandas._libs.lib import is_list_like
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import notna
def _convert_listlike(arg, format):
    if isinstance(arg, (list, tuple)):
        arg = np.array(arg, dtype='O')
    elif getattr(arg, 'ndim', 1) > 1:
        raise TypeError('arg must be a string, datetime, list, tuple, 1-d array, or Series')
    arg = np.asarray(arg, dtype='O')
    if infer_time_format and format is None:
        format = _guess_time_format_for_array(arg)
    times: list[time | None] = []
    if format is not None:
        for element in arg:
            try:
                times.append(datetime.strptime(element, format).time())
            except (ValueError, TypeError) as err:
                if errors == 'raise':
                    msg = f'Cannot convert {element} to a time with given format {format}'
                    raise ValueError(msg) from err
                if errors == 'ignore':
                    return arg
                else:
                    times.append(None)
    else:
        formats = _time_formats[:]
        format_found = False
        for element in arg:
            time_object = None
            try:
                time_object = time.fromisoformat(element)
            except (ValueError, TypeError):
                for time_format in formats:
                    try:
                        time_object = datetime.strptime(element, time_format).time()
                        if not format_found:
                            fmt = formats.pop(formats.index(time_format))
                            formats.insert(0, fmt)
                            format_found = True
                        break
                    except (ValueError, TypeError):
                        continue
            if time_object is not None:
                times.append(time_object)
            elif errors == 'raise':
                raise ValueError(f'Cannot convert arg {arg} to a time')
            elif errors == 'ignore':
                return arg
            else:
                times.append(None)
    return times