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
class QualificationRequest(BaseAutoResultElement):
    """
    Class to extract an QualificationRequest structure from a response (used in
    ResultSet)

    Will have attributes named as per the Developer Guide,
    e.g. QualificationRequestId, QualificationTypeId, SubjectId, etc
    """

    def __init__(self, connection):
        super(QualificationRequest, self).__init__(connection)
        self.answers = []

    def endElement(self, name, value, connection):
        if name == 'Answer':
            answer_rs = ResultSet([('Answer', QuestionFormAnswer)])
            h = handler.XmlHandler(answer_rs, connection)
            value = connection.get_utf8able_str(value)
            xml.sax.parseString(value, h)
            self.answers.append(answer_rs)
        else:
            super(QualificationRequest, self).endElement(name, value, connection)