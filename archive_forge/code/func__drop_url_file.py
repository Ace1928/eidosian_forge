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
def _drop_url_file(self, url, conditional):
    filename = url.database
    if filename and (not conditional or os.access(filename, os.F_OK)):
        os.remove(filename)