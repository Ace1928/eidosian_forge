from oslo_utils import reflection
from neutron_lib._i18n import _
from neutron_lib import exceptions
class NeutronDbObjectDuplicateEntry(exceptions.Conflict):
    """Duplicate entry at unique column error.

    Raised when made an attempt to write to a unique column the same entry as
    existing one. :attr: `columns` available on an instance of the exception
    and could be used at error handling::

       try:
           obj_ref.save()
       except NeutronDbObjectDuplicateEntry as e:
           if 'colname' in e.columns:
               # Handle error.
    """
    message = _('Failed to create a duplicate %(object_type)s: for attribute(s) %(attributes)s with value(s) %(values)s')

    def __init__(self, object_class, db_exception):
        super().__init__(object_type=reflection.get_class_name(object_class, fully_qualified=False), attributes=db_exception.columns, values=db_exception.value)
        self.columns = db_exception.columns
        self.value = db_exception.value