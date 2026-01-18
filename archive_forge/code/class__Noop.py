from oslo_config import cfg
from oslo_log import log as logging
from taskflow.patterns import linear_flow as lf
from taskflow import task
class _Noop(task.Task):

    def __init__(self, task_id, task_type, image_repo):
        self.task_id = task_id
        self.task_type = task_type
        self.image_repo = image_repo
        super(_Noop, self).__init__(name='%s-Noop-%s' % (task_type, task_id))

    def execute(self, **kwargs):
        LOG.debug('No_op import plugin')
        return

    def revert(self, result=None, **kwargs):
        if result is not None:
            LOG.debug('No_op import plugin failed')
            return