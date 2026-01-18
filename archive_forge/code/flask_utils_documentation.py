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

    Convenience wrapper for configuring a Peewee database for use with a Flask
    application. Provides a base `Model` class and registers handlers to manage
    the database connection during the request/response cycle.

    Usage::

        from flask import Flask
        from peewee import *
        from playhouse.flask_utils import FlaskDB


        # The database can be specified using a database URL, or you can pass a
        # Peewee database instance directly:
        DATABASE = 'postgresql:///my_app'
        DATABASE = PostgresqlDatabase('my_app')

        # If we do not want connection-management on any views, we can specify
        # the view names using FLASKDB_EXCLUDED_ROUTES. The db connection will
        # not be opened/closed automatically when these views are requested:
        FLASKDB_EXCLUDED_ROUTES = ('logout',)

        app = Flask(__name__)
        app.config.from_object(__name__)

        # Now we can configure our FlaskDB:
        flask_db = FlaskDB(app)

        # Or use the "deferred initialization" pattern:
        flask_db = FlaskDB()
        flask_db.init_app(app)

        # The `flask_db` provides a base Model-class for easily binding models
        # to the configured database:
        class User(flask_db.Model):
            email = CharField()

    