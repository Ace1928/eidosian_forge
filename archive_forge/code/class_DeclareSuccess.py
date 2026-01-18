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
class DeclareSuccess(task.Task):

    def execute(self, sent_to):
        print('Done!')
        print('All data processed and sent to %s' % sent_to)