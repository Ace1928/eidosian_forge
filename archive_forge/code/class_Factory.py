import logging
import sys
import textwrap
from absl import app
from absl import flags
import httplib2
import termcolor
import bigquery_client
import bq_auth_flags
import bq_flags
import bq_utils
import credential_loader
from auth import main_credential_loader
from frontend import utils as bq_frontend_utils
from utils import bq_logging
class Factory(object):
    """Class encapsulating factory creation of BigqueryClient."""
    _BIGQUERY_CLIENT_FACTORY = None

    class ClientTablePrinter(object):
        """Class encapsulating factory creation of TablePrinter."""
        _TABLE_PRINTER = None

        @classmethod
        def GetTablePrinter(cls):
            if cls._TABLE_PRINTER is None:
                cls._TABLE_PRINTER = bq_frontend_utils.TablePrinter()
            return cls._TABLE_PRINTER

        @classmethod
        def SetTablePrinter(cls, printer):
            if not isinstance(printer, bq_frontend_utils.TablePrinter):
                raise TypeError('Printer must be an instance of TablePrinter.')
            cls._TABLE_PRINTER = printer

    @classmethod
    def GetBigqueryClientFactory(cls):
        if cls._BIGQUERY_CLIENT_FACTORY is None:
            cls._BIGQUERY_CLIENT_FACTORY = bigquery_client.BigqueryClient
        return cls._BIGQUERY_CLIENT_FACTORY

    @classmethod
    def SetBigqueryClientFactory(cls, factory):
        if not issubclass(factory, bigquery_client.BigqueryClient):
            raise TypeError('Factory must be subclass of BigqueryClient.')
        cls._BIGQUERY_CLIENT_FACTORY = factory