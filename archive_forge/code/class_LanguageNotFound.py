import gettext
import logging
import os
import sqlite3
import sys
class LanguageNotFound(Exception):
    """
    The specified language wasn't found in the database.
    """