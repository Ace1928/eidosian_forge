import math
import sys
from flask import abort
from flask import render_template
from flask import request
from peewee import Database
from peewee import DoesNotExist
from peewee import Model
from peewee import Proxy
from peewee import SelectQuery
from playhouse.db_url import connect as db_url_connect
def _load_database(self, app, config_value):
    if isinstance(config_value, Database):
        database = config_value
    elif isinstance(config_value, dict):
        database = self._load_from_config_dict(dict(config_value))
    else:
        database = db_url_connect(config_value)
    if isinstance(self.database, Proxy):
        self.database.initialize(database)
    else:
        self.database = database