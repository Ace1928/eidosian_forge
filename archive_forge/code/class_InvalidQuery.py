import pandas
import pandas._libs.lib as lib
from sqlalchemy import MetaData, Table, create_engine, inspect, text
from modin.core.storage_formats.pandas.parsers import _split_result_for_readers
class InvalidQuery(Exception):
    """Exception that should be raised if invalid query statement was found."""