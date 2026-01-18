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
class QuestionFormAnswer(BaseAutoResultElement):
    """
    Class to extract Answers from inside the embedded XML
    QuestionFormAnswers element inside the Answer element which is
    part of the Assignment and QualificationRequest structures

    A QuestionFormAnswers element contains an Answer element for each
    question in the HIT or Qualification test for which the Worker
    provided an answer. Each Answer contains a QuestionIdentifier
    element whose value corresponds to the QuestionIdentifier of a
    Question in the QuestionForm. See the QuestionForm data structure
    for more information about questions and answer specifications.

    If the question expects a free-text answer, the Answer element
    contains a FreeText element. This element contains the Worker's
    answer

    *NOTE* - currently really only supports free-text and selection answers
    """

    def __init__(self, connection):
        super(QuestionFormAnswer, self).__init__(connection)
        self.fields = []
        self.qid = None

    def endElement(self, name, value, connection):
        if name == 'QuestionIdentifier':
            self.qid = value
        elif name in ['FreeText', 'SelectionIdentifier', 'OtherSelectionText'] and self.qid:
            self.fields.append(value)