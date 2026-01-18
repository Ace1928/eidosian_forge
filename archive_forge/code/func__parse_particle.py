from typing import Any, Optional, Tuple, Union
from ..exceptions import XMLSchemaValueError
from ..aliases import ElementType, ModelParticleType
from ..translation import gettext as _
def _parse_particle(self, elem: ElementType) -> None:
    if 'minOccurs' in elem.attrib:
        try:
            min_occurs = int(elem.attrib['minOccurs'])
        except (TypeError, ValueError):
            msg = _('minOccurs value is not an integer value')
            self.parse_error(msg)
        else:
            if min_occurs < 0:
                msg = _('minOccurs value must be a non negative integer')
                self.parse_error(msg)
            else:
                self.min_occurs = min_occurs
    max_occurs = elem.get('maxOccurs')
    if max_occurs is None:
        if self.min_occurs > 1:
            msg = _('minOccurs must be lesser or equal than maxOccurs')
            self.parse_error(msg)
    elif max_occurs == 'unbounded':
        self.max_occurs = None
    else:
        try:
            self.max_occurs = int(max_occurs)
        except ValueError:
            msg = _("maxOccurs value must be a non negative integer or 'unbounded'")
            self.parse_error(msg)
        else:
            if self.min_occurs > self.max_occurs:
                msg = _("maxOccurs must be 'unbounded' or greater than minOccurs")
                self.parse_error(msg)
                self.max_occurs = None