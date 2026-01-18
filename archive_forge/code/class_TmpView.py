from collections import abc
import re
from oslotest import base
from oslo_reports.models import base as base_model
from oslo_reports import report
class TmpView(object):

    def __call__(self, model):
        return '{len: ' + str(len(model.c)) + '}'