import copy
import logging
from functools import total_ordering
from tornado import web
from ..utils.tasks import as_dict, get_task_by_id, iter_tasks
from ..views import BaseHandler
class TasksView(BaseHandler):

    @web.authenticated
    def get(self):
        app = self.application
        capp = self.application.capp
        time = 'natural-time' if app.options.natural_time else 'time'
        if capp.conf.timezone:
            time += '-' + str(capp.conf.timezone)
        self.render('tasks.html', tasks=[], columns=app.options.tasks_columns, time=time)