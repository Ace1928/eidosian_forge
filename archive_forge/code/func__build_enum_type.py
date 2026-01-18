from . import model
from .commontypes import COMMON_TYPES, resolve_common_type
from .error import FFIError, CDefError
import weakref, re, sys
def _build_enum_type(self, explicit_name, decls):
    if decls is not None:
        partial = False
        enumerators = []
        enumvalues = []
        nextenumvalue = 0
        for enum in decls.enumerators:
            if _r_enum_dotdotdot.match(enum.name):
                partial = True
                continue
            if enum.value is not None:
                nextenumvalue = self._parse_constant(enum.value)
            enumerators.append(enum.name)
            enumvalues.append(nextenumvalue)
            self._add_constants(enum.name, nextenumvalue)
            nextenumvalue += 1
        enumerators = tuple(enumerators)
        enumvalues = tuple(enumvalues)
        tp = model.EnumType(explicit_name, enumerators, enumvalues)
        tp.partial = partial
    else:
        tp = model.EnumType(explicit_name, (), ())
    return tp