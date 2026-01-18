from __future__ import absolute_import, division, print_function
import copy
from functools import partial
from ansible_collections.community.network.plugins.module_utils.network.ftd.common import HTTPMethod, equal_objects, FtdConfigurationError, \
from ansible_collections.community.network.plugins.module_utils.network.ftd.fdm_swagger_client import OperationField, ValidationError
from ansible.module_utils.six import iteritems
class OperationChecker(object):

    @classmethod
    def is_add_operation(cls, operation_name, operation_spec):
        """
        Check if operation defined with 'operation_name' is add object operation according to 'operation_spec'.

        :param operation_name: name of the operation being called by the user
        :type operation_name: str
        :param operation_spec: specification of the operation being called by the user
        :type operation_spec: dict
        :return: True if the called operation is add object operation, otherwise False
        :rtype: bool
        """
        return operation_name.startswith(OperationNamePrefix.ADD) and is_post_request(operation_spec)

    @classmethod
    def is_edit_operation(cls, operation_name, operation_spec):
        """
        Check if operation defined with 'operation_name' is edit object operation according to 'operation_spec'.

        :param operation_name: name of the operation being called by the user
        :type operation_name: str
        :param operation_spec: specification of the operation being called by the user
        :type operation_spec: dict
        :return: True if the called operation is edit object operation, otherwise False
        :rtype: bool
        """
        return operation_name.startswith(OperationNamePrefix.EDIT) and is_put_request(operation_spec)

    @classmethod
    def is_delete_operation(cls, operation_name, operation_spec):
        """
        Check if operation defined with 'operation_name' is delete object operation according to 'operation_spec'.

        :param operation_name: name of the operation being called by the user
        :type operation_name: str
        :param operation_spec: specification of the operation being called by the user
        :type operation_spec: dict
        :return: True if the called operation is delete object operation, otherwise False
        :rtype: bool
        """
        return operation_name.startswith(OperationNamePrefix.DELETE) and operation_spec[OperationField.METHOD] == HTTPMethod.DELETE

    @classmethod
    def is_get_list_operation(cls, operation_name, operation_spec):
        """
        Check if operation defined with 'operation_name' is get list of objects operation according to 'operation_spec'.

        :param operation_name: name of the operation being called by the user
        :type operation_name: str
        :param operation_spec: specification of the operation being called by the user
        :type operation_spec: dict
        :return: True if the called operation is get a list of objects operation, otherwise False
        :rtype: bool
        """
        return operation_spec[OperationField.METHOD] == HTTPMethod.GET and operation_spec[OperationField.RETURN_MULTIPLE_ITEMS]

    @classmethod
    def is_get_operation(cls, operation_name, operation_spec):
        """
        Check if operation defined with 'operation_name' is get objects operation according to 'operation_spec'.

        :param operation_name: name of the operation being called by the user
        :type operation_name: str
        :param operation_spec: specification of the operation being called by the user
        :type operation_spec: dict
        :return: True if the called operation is get object operation, otherwise False
        :rtype: bool
        """
        return operation_spec[OperationField.METHOD] == HTTPMethod.GET and (not operation_spec[OperationField.RETURN_MULTIPLE_ITEMS])

    @classmethod
    def is_upsert_operation(cls, operation_name):
        """
        Check if operation defined with 'operation_name' is upsert objects operation according to 'operation_name'.

        :param operation_name: name of the operation being called by the user
        :type operation_name: str
        :return: True if the called operation is upsert object operation, otherwise False
        :rtype: bool
        """
        return operation_name.startswith(OperationNamePrefix.UPSERT)

    @classmethod
    def is_find_by_filter_operation(cls, operation_name, params, operation_spec):
        """
        Checks whether the called operation is 'find by filter'. This operation fetches all objects and finds
        the matching ones by the given filter. As filtering is done on the client side, this operation should be used
        only when selected filters are not implemented on the server side.

        :param operation_name: name of the operation being called by the user
        :type operation_name: str
        :param operation_spec: specification of the operation being called by the user
        :type operation_spec: dict
        :param params: params - params should contain 'filters'
        :return: True if the called operation is find by filter, otherwise False
        :rtype: bool
        """
        is_get_list = cls.is_get_list_operation(operation_name, operation_spec)
        return is_get_list and ParamName.FILTERS in params and params[ParamName.FILTERS]

    @classmethod
    def is_upsert_operation_supported(cls, operations):
        """
        Checks if all operations required for upsert object operation are defined in 'operations'.

        :param operations: specification of the operations supported by model
        :type operations: dict
        :return: True if all criteria required to provide requested called operation are satisfied, otherwise False
        :rtype: bool
        """
        has_edit_op = next((name for name, spec in iteritems(operations) if cls.is_edit_operation(name, spec)), None)
        has_get_list_op = next((name for name, spec in iteritems(operations) if cls.is_get_list_operation(name, spec)), None)
        return has_edit_op and has_get_list_op