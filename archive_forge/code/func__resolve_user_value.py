import collections
from oslo_serialization import jsonutils
from heat.common import exception
from heat.common.i18n import _
from heat.common import param_utils
from heat.engine import constraints as constr
from heat.engine import function
from heat.engine.hot import parameters as hot_param
from heat.engine import parameters
from heat.engine import support
from heat.engine import translation as trans
def _resolve_user_value(self, key, prop, validate):
    """Return the user-supplied value (or None), and whether it was found.

        This allows us to distinguish between, on the one hand, either a
        Function that returns None or an explicit null value passed and, on the
        other hand, either no value passed or a Macro that returns Ellipsis,
        meaning that the result should be treated the same as if no value were
        passed.
        """
    if key not in self.data:
        return (None, False)
    if self.translation.is_deleted(prop.path) or self.translation.is_replaced(prop.path):
        return (None, False)
    try:
        unresolved_value = self.data[key]
        if validate:
            if self._find_deps_any_in_init(unresolved_value):
                validate = False
        value = self.resolve(unresolved_value, nullable=True)
        if value is Ellipsis:
            return (None, False)
        if self.translation.has_translation(prop.path):
            value = self.translation.translate(prop.path, value, self.data)
        return (prop.get_value(value, validate, translation=self.translation), True)
    except exception.StackValidationFailed as e:
        raise exception.StackValidationFailed(path=e.path, message=e.error_message)
    except Exception as e:
        raise ValueError(str(e))