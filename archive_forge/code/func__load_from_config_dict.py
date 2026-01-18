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
def _load_from_config_dict(self, config_dict):
    try:
        name = config_dict.pop('name')
        engine = config_dict.pop('engine')
    except KeyError:
        raise RuntimeError('DATABASE configuration must specify a `name` and `engine`.')
    if '.' in engine:
        path, class_name = engine.rsplit('.', 1)
    else:
        path, class_name = ('peewee', engine)
    try:
        __import__(path)
        module = sys.modules[path]
        database_class = getattr(module, class_name)
        assert issubclass(database_class, Database)
    except ImportError:
        raise RuntimeError('Unable to import %s' % engine)
    except AttributeError:
        raise RuntimeError('Database engine not found %s' % engine)
    except AssertionError:
        raise RuntimeError('Database engine not a subclass of peewee.Database: %s' % engine)
    return database_class(name, **config_dict)