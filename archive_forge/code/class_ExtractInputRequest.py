import json
import logging
import os
import sys
import time
from oslo_utils import uuidutils
from taskflow import engines
from taskflow.listeners import printing
from taskflow.patterns import graph_flow as gf
from taskflow.patterns import linear_flow as lf
from taskflow import task
from taskflow.utils import misc
class ExtractInputRequest(task.Task):

    def __init__(self, resources):
        super(ExtractInputRequest, self).__init__(provides='parsed_request')
        self._resources = resources

    def execute(self, request):
        return {'user': request.user, 'user_id': misc.as_int(request.id), 'request_id': uuidutils.generate_uuid()}