import csv
import logging
import os
import random
import sys
import futurist
from taskflow import engines
from taskflow.patterns import unordered_flow as uf
from taskflow import task
class RowMultiplier(task.Task):
    """Performs a modification of an input row, creating a output row."""

    def __init__(self, name, index, row, multiplier):
        super(RowMultiplier, self).__init__(name=name)
        self.index = index
        self.multiplier = multiplier
        self.row = row

    def execute(self):
        return [r * self.multiplier for r in self.row]