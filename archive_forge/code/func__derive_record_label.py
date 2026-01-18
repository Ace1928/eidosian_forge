import datetime
import logging
from lxml import etree
import io
import warnings
import prov
import prov.identifier
from prov.model import DEFAULT_NAMESPACES, sorted_attributes
from prov.constants import *  # NOQA
from prov.serializers import Serializer
def _derive_record_label(self, rec_type, attributes):
    """
        Helper function trying to derive the record label taking care of
        subtypes and what not. It will also remove the type declaration for
        the attributes if it was used to specialize the type.

        :param rec_type: The type of records.
        :param attributes: The attributes of the record.
        """
    rec_label = FULL_NAMES_MAP[rec_type]
    for key, value in list(attributes):
        if key != PROV_TYPE:
            continue
        if isinstance(value, prov.model.Literal):
            value = value.value
        if value in PROV_BASE_CLS and PROV_BASE_CLS[value] != value:
            attributes.remove((key, value))
            rec_label = FULL_NAMES_MAP[value]
            break
    return rec_label