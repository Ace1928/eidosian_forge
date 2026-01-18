import warnings
from django.core import exceptions
from django.utils.deprecation import RemovedInDjango60Warning
from django.utils.functional import cached_property
from django.utils.hashable import make_hashable
from . import BLANK_CHOICE_DASH
from .mixins import FieldCacheMixin
class OneToOneRel(ManyToOneRel):
    """
    Used by OneToOneField to store information about the relation.

    ``_meta.get_fields()`` returns this class to provide access to the field
    flags for the reverse relation.
    """

    def __init__(self, field, to, field_name, related_name=None, related_query_name=None, limit_choices_to=None, parent_link=False, on_delete=None):
        super().__init__(field, to, field_name, related_name=related_name, related_query_name=related_query_name, limit_choices_to=limit_choices_to, parent_link=parent_link, on_delete=on_delete)
        self.multiple = False