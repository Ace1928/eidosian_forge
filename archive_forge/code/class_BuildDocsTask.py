import logging
import os
import sys
import taskflow.engines
from taskflow.patterns import graph_flow as gf
from taskflow import task
import example_utils as eu  # noqa
class BuildDocsTask(task.Task):
    """Pretends to build docs from sources."""
    default_provides = 'docs'

    def execute(self, **kwargs):
        for source_filename in kwargs.values():
            print('Building docs for %s' % source_filename)
        return 'docs'