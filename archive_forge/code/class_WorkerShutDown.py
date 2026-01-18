import logging
from tornado import web
from . import BaseApiHandler
class WorkerShutDown(ControlHandler):

    @web.authenticated
    def post(self, workername):
        """
Shut down a worker

**Example request**:

.. sourcecode:: http

  POST /api/worker/shutdown/celery@worker2 HTTP/1.1
  Content-Length: 0
  Host: localhost:5555

**Example response**:

.. sourcecode:: http

  HTTP/1.1 200 OK
  Content-Length: 29
  Content-Type: application/json; charset=UTF-8

  {
      "message": "Shutting down!"
  }

:reqheader Authorization: optional OAuth token to authenticate
:statuscode 200: no error
:statuscode 401: unauthorized request
:statuscode 404: unknown worker
        """
        if not self.is_worker(workername):
            raise web.HTTPError(404, f"Unknown worker '{workername}'")
        logger.info("Shutting down '%s' worker", workername)
        self.capp.control.broadcast('shutdown', destination=[workername])
        self.write(dict(message='Shutting down!'))