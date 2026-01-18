import getpass
import inspect
import os
import sys
import textwrap
import decorator
from magnumclient.common.apiclient import exceptions
from oslo_utils import encodeutils
from oslo_utils import strutils
import prettytable
from magnumclient.i18n import _
def _get_list_table_columns_and_formatters(fields, objs, exclude_fields=(), filters=None):
    """Check and add fields to output columns.

    If there is any value in fields that not an attribute of obj,
    CommandError will be raised.
    If fields has duplicate values (case sensitive), we will make them unique
    and ignore duplicate ones.
    :param fields: A list of string contains the fields to be printed.
    :param objs: An list of object which will be used to check if field is
                 valid or not. Note, we don't check fields if obj is None or
                 empty.
    :param exclude_fields: A tuple of string which contains the fields to be
                           excluded.
    :param filters: A dictionary defines how to get value from fields, this
                    is useful when field's value is a complex object such as
                    dictionary.
    :return: columns, formatters.
             columns is a list of string which will be used as table header.
             formatters is a dictionary specifies how to display the value
             of the field.
             They can be [], {}.
    :raise: magnumclient.common.apiclient.exceptions.CommandError.
    """
    if objs and isinstance(objs, list):
        obj = objs[0]
    else:
        obj = None
        fields = None
    columns = []
    formatters = {}
    if fields:
        non_existent_fields = []
        exclude_fields = set(exclude_fields)
        for field in fields.split(','):
            if not hasattr(obj, field):
                non_existent_fields.append(field)
                continue
            if field in exclude_fields:
                continue
            field_title, formatter = make_field_formatter(field, filters)
            columns.append(field_title)
            formatters[field_title] = formatter
            exclude_fields.add(field)
        if non_existent_fields:
            raise exceptions.CommandError(_('Non-existent fields are specified: %s') % non_existent_fields)
    return (columns, formatters)