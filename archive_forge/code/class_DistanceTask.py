import collections
import math
import os
import sys
from taskflow import engines
from taskflow.patterns import linear_flow
from taskflow import task
class DistanceTask(task.Task):
    default_provides = 'distance'

    def execute(self, a=Point(0, 0), b=Point(0, 0)):
        return math.sqrt(math.pow(b.x - a.x, 2) + math.pow(b.y - a.y, 2))