from oslo_log import log as logging
import sqlalchemy.types
def Boolean():
    return sqlalchemy.types.Boolean(create_constraint=True, name=None)