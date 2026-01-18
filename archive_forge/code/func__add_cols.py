from sqlalchemy import Column
from sqlalchemy import event
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import Table
from sqlalchemy.sql import text
from ...testing.fixtures import AlterColRoundTripFixture
from ...testing.fixtures import TestBase
@event.listens_for(Table, 'after_parent_attach')
def _add_cols(table, metadata):
    if table.name == 'tbl_with_auto_appended_column':
        table.append_column(Column('bat', Integer))