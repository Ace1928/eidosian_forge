import logging
from alembic import op
import sqlalchemy as sa
from taskflow.persistence.backends.sqlalchemy import tables
def _get_indexes():
    indexes = [{'index_name': 'logbook_uuid_idx', 'table_name': 'logbooks', 'columns': ['uuid']}, {'index_name': 'flowdetails_uuid_idx', 'table_name': 'flowdetails', 'columns': ['uuid']}, {'index_name': 'taskdetails_uuid_idx', 'table_name': 'taskdetails', 'columns': ['uuid']}]
    return indexes