import datetime
import textwrap
from alembic import op
from oslo_log import log
import sqlalchemy as sql
from keystone.assignment.backends import sql as assignment_sql
from keystone.common import sql as ks_sql
import keystone.conf
from keystone.identity.mapping_backends import mapping as mapping_backend
Initial version.

Revision ID: 27e647c0fad4
Revises:
Create Date: 2021-12-23 11:13:26.305412
