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
class MakeDBEntry(task.Task):

    def __init__(self, resources):
        super(MakeDBEntry, self).__init__()
        self._resources = resources

    def execute(self, parsed_request):
        db_handle = self._resources.db_handle
        db_handle.query('INSERT %s INTO mydb' % parsed_request)

    def revert(self, result, parsed_request):
        db_handle = self._resources.db_handle
        db_handle.query('DELETE %s FROM mydb IF EXISTS' % parsed_request)