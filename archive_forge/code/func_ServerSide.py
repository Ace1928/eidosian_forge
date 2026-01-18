import json
import logging
import uuid
from peewee import *
from peewee import ColumnBase
from peewee import Expression
from peewee import Node
from peewee import NodeList
from peewee import __deprecated__
def ServerSide(query, database=None, array_size=None):
    if database is None:
        database = query._database
    with database.transaction():
        server_side_query = ServerSideQuery(query, array_size=array_size)
        for row in server_side_query:
            yield row