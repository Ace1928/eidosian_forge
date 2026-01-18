import logging
from alembic import op
import sqlalchemy as sa
from taskflow.persistence.backends.sqlalchemy import tables
def _get_foreign_keys():
    f_keys = [{'constraint_name': 'flowdetails_ibfk_1', 'source_table': 'flowdetails', 'referent_table': 'logbooks', 'local_cols': ['parent_uuid'], 'remote_cols': ['uuid'], 'ondelete': 'CASCADE'}, {'constraint_name': 'taskdetails_ibfk_1', 'source_table': 'taskdetails', 'referent_table': 'flowdetails', 'local_cols': ['parent_uuid'], 'remote_cols': ['uuid'], 'ondelete': 'CASCADE'}]
    return f_keys