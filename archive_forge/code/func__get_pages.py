import xml.sax
import datetime
import itertools
from boto import handler
from boto import config
from boto.mturk.price import Price
import boto.mturk.notification
from boto.connection import AWSQueryConnection
from boto.exception import EC2ResponseError
from boto.resultset import ResultSet
from boto.mturk.question import QuestionForm, ExternalQuestion, HTMLQuestion
@staticmethod
def _get_pages(page_size, total_records):
    """
        Given a page size (records per page) and a total number of
        records, return the page numbers to be retrieved.
        """
    pages = total_records / page_size + bool(total_records % page_size)
    return list(range(1, pages + 1))