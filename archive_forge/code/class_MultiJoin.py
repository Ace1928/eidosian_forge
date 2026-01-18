import warnings
from django.core.exceptions import FullResultSet
from django.db.models.sql.constants import INNER, LOUTER
from django.utils.deprecation import RemovedInDjango60Warning
class MultiJoin(Exception):
    """
    Used by join construction code to indicate the point at which a
    multi-valued join was attempted (if the caller wants to treat that
    exceptionally).
    """

    def __init__(self, names_pos, path_with_names):
        self.level = names_pos
        self.names_with_path = path_with_names