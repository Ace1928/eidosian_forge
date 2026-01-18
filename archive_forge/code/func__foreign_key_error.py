import collections
import logging
import re
import sys
from sqlalchemy import event
from sqlalchemy import exc as sqla_exc
from oslo_db import exception
from oslo_db.sqlalchemy import compat
@filters('sqlite', sqla_exc.IntegrityError, '(?i).*foreign key constraint failed')
@filters('postgresql', sqla_exc.IntegrityError, '.*on table \\"(?P<table>[^\\"]+)\\" violates foreign key constraint \\"(?P<constraint>[^\\"]+)\\".*\\nDETAIL:  Key \\((?P<key>.+)\\)=\\(.+\\) is (not present in|still referenced from) table \\"(?P<key_table>[^\\"]+)\\".')
@filters('mysql', sqla_exc.IntegrityError, '.*Cannot (add|delete) or update a (child|parent) row: a foreign key constraint fails \\([`"].+[`"]\\.[`"](?P<table>.+)[`"], CONSTRAINT [`"](?P<constraint>.+)[`"] FOREIGN KEY \\([`"](?P<key>.+)[`"]\\) REFERENCES [`"](?P<key_table>.+)[`"] ')
def _foreign_key_error(integrity_error, match, engine_name, is_disconnect):
    """Filter for foreign key errors."""
    try:
        table = match.group('table')
    except IndexError:
        table = None
    try:
        constraint = match.group('constraint')
    except IndexError:
        constraint = None
    try:
        key = match.group('key')
    except IndexError:
        key = None
    try:
        key_table = match.group('key_table')
    except IndexError:
        key_table = None
    raise exception.DBReferenceError(table, constraint, key, key_table, integrity_error)