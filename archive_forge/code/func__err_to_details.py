import csv
import datetime
import testtools
from testtools import StreamResult
from testtools.content import TracebackContent, text_content
import iso8601
import subunit
def _err_to_details(self, test, err, details):
    if details:
        return details
    return {'traceback': TracebackContent(err, test)}