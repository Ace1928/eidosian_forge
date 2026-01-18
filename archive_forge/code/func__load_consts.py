import os
from functools import total_ordering
from ._clib import Libevdev
import libevdev
def _load_consts():
    """
    Loads all event type, code and property names and makes them available
    as enums in the module. Use as e.g. libevdev.EV_SYN.SYN_REPORT.

    Available are::

    libevdev.types ... an list containing all event types, e.g.
                         libevdev.EV_TYPES.EV_REL

    libevdev.EV_REL ... an enum containing all REL event types, e.g.
                        libevdev.EV_REL.REL_X. The name of each enum value
                        is the string of the code ('REL_X'), the value is the integer
                        value of that code.

    libevdev.EV_ABS ... as above, but for EV_ABS

    libevdev.EV_BITS ... libevdev.EV_FOO as an enum

    Special attributes are (an apply to all EV_foo enums):
        libevdev.EV_REL.type ... the EV_TYPES entry of the event type
        libevdev.EV_REL.max  ... the maximum code in this event type
    """
    Libevdev()
    tmax = Libevdev.event_to_value('EV_MAX')
    assert tmax is not None
    types = []
    for t in range(tmax + 1):
        tname = Libevdev.event_to_name(t)
        if tname is None:
            continue
        cmax = Libevdev.type_max(t)
        new_class = type(tname, (EventType,), {'value': t, 'name': tname, 'max': cmax})
        type_object = new_class()
        setattr(libevdev, tname, type_object)
        types.append(type_object)
        if cmax is None:
            setattr(type_object, 'codes', [])
            continue
        codes = []
        for c in range(cmax + 1):
            cname = Libevdev.event_to_name(t, c)
            name = cname
            has_name = cname is not None
            if cname is None:
                name = '{}_{:02X}'.format(tname[3:], c)
                cname = '_{}'.format(name)
            new_class = type(cname, (EventCode,), {'type': type_object, 'name': name, 'value': c, 'is_defined': has_name})
            code_object = new_class()
            setattr(type_object, cname, code_object)
            codes.append(code_object)
        setattr(type_object, 'codes', codes)
    setattr(libevdev, 'types', types)
    pmax = Libevdev.property_to_value('INPUT_PROP_MAX')
    assert pmax is not None
    props = []
    for p in range(pmax + 1):
        pname = Libevdev.property_to_name(p)
        if pname is None:
            continue
        new_class = type(pname, (InputProperty,), {'value': p, 'name': pname})
        prop_object = new_class()
        setattr(libevdev, pname, prop_object)
        props.append(prop_object)
    setattr(libevdev, 'props', props)