import abc
import logging
import os
import random
import re
import string
import sqlalchemy
from sqlalchemy import schema
from sqlalchemy import sql
import testresources
from oslo_db import exception
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import utils
def _random_ident():
    return ''.join((random.choice(string.ascii_lowercase) for i in range(10)))