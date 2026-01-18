import logging
from tornado import web
from . import BaseApiHandler
class TaskRateLimit(ControlHandler):

    @web.authenticated
    def post(self, taskname):
        """
Change rate limit for a task

**Example request**:

.. sourcecode:: http

    POST /api/task/rate-limit/tasks.sleep HTTP/1.1
    Content-Length: 41
    Content-Type: application/x-www-form-urlencoded; charset=utf-8
    Host: localhost:5555

    ratelimit=200&workername=celery%40worker1

**Example response**:

.. sourcecode:: http

  HTTP/1.1 200 OK
  Content-Length: 61
  Content-Type: application/json; charset=UTF-8

  {
      "message": "new rate limit set successfully"
  }

:query workername: worker name
:reqheader Authorization: optional OAuth token to authenticate
:statuscode 200: no error
:statuscode 401: unauthorized request
:statuscode 404: unknown task/worker
        """
        workername = self.get_argument('workername')
        ratelimit = self.get_argument('ratelimit')
        if taskname not in self.capp.tasks:
            raise web.HTTPError(404, f"Unknown task '{taskname}'")
        if workername is not None and (not self.is_worker(workername)):
            raise web.HTTPError(404, f"Unknown worker '{workername}'")
        logger.info("Setting '%s' rate limit for '%s' task", ratelimit, taskname)
        destination = [workername] if workername is not None else None
        response = self.capp.control.rate_limit(taskname, ratelimit, reply=True, destination=destination)
        if response and 'ok' in response[0][workername]:
            self.write(dict(message=response[0][workername]['ok']))
        else:
            logger.error(response)
            self.set_status(403)
            reason = self.error_reason(taskname, response)
            self.write(f"Failed to set rate limit: '{reason}'")