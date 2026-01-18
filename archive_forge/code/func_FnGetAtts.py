import itertools
from heat.common import exception
from heat.engine import attributes
from heat.engine import status
def FnGetAtts(self):
    """For the intrinsic function get_attr when getting all attributes.

        :returns: a dict of all of the resource's attribute values, excluding
                  the "show" attribute.
        """
    all_attrs = self._res_data().attributes()
    return dict(((k, v) for k, v in all_attrs.items() if k != attributes.SHOW_ATTR))