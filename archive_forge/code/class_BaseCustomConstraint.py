import collections
import json
import numbers
import re
from oslo_cache import core
from oslo_config import cfg
from oslo_log import log
from oslo_utils import reflection
from oslo_utils import strutils
from heat.common import cache
from heat.common import exception
from heat.common.i18n import _
from heat.engine import resources
class BaseCustomConstraint(object):
    """A base class for validation using API clients.

    It will provide a better error message, and reduce a bit of duplication.
    Subclass must provide `expected_exceptions` and implement
    `validate_with_client`.
    """
    expected_exceptions = (exception.EntityNotFound,)
    resource_client_name = None
    resource_getter_name = None
    _error_message = None

    def error(self, value):
        if self._error_message is None:
            return _("Error validating value '%(value)s'") % {'value': value}
        return _("Error validating value '%(value)s': %(message)s") % {'value': value, 'message': self._error_message}

    def validate(self, value, context):

        @MEMOIZE
        def check_cache_or_validate_value(cache_value_prefix, value_to_validate):
            """Check if validation result stored in cache or validate value.

            The function checks that value was validated and validation
            result stored in cache. If not then it executes validation and
            stores the result of validation in cache.
            If caching is disabled it requests for validation each time.

            :param cache_value_prefix: cache prefix that used to distinguish
                                       value in heat cache. So the cache key
                                       would be the following:
                                       cache_value_prefix + value_to_validate.
            :param value_to_validate: value that need to be validated
            :return: True if value is valid otherwise False
            """
            try:
                self.validate_with_client(context.clients, value_to_validate)
            except self.expected_exceptions as e:
                self._error_message = str(e)
                return False
            else:
                return True
        class_name = reflection.get_class_name(self, fully_qualified=False)
        cache_value_prefix = '{0}:{1}'.format(class_name, str(context.tenant_id))
        validation_result = check_cache_or_validate_value(cache_value_prefix, value)
        if not validation_result:
            check_cache_or_validate_value.invalidate(cache_value_prefix, value)
        return validation_result

    def validate_with_client(self, client, resource_id):
        if self.resource_client_name and self.resource_getter_name:
            getattr(client.client_plugin(self.resource_client_name), self.resource_getter_name)(resource_id)
        else:
            raise exception.InvalidSchemaError(message=_('Client name and resource getter name must be specified.'))