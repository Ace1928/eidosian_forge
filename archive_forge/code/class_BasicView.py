from collections import abc
import re
from oslotest import base
from oslo_reports.models import base as base_model
from oslo_reports import report
class BasicView(object):

    def __call__(self, model):
        res = ''
        for k in sorted(model.keys()):
            res += str(k) + ': ' + str(model[k]) + ';'
        return res