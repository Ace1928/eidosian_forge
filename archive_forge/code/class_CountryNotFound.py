import gettext
import logging
import os
import sqlite3
import sys
class CountryNotFound(Exception):
    """
    The specified country wasn't found in the database.
    """