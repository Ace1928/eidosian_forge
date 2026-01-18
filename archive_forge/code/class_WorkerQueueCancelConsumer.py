import logging
from tornado import web
from . import BaseApiHandler
class WorkerQueueCancelConsumer(ControlHandler):

    @web.authenticated
    def post(self, workername):
        """
Stop consuming from a queue

**Example request**:

.. sourcecode:: http

  POST /api/worker/queue/cancel-consumer/celery@worker2?queue=sample-queue
  Content-Length: 0
  Content-Type: application/x-www-form-urlencoded; charset=utf-8
  Host: localhost:5555

**Example response**:

.. sourcecode:: http

  HTTP/1.1 200 OK
  Content-Length: 52
  Content-Type: application/json; charset=UTF-8

  {
      "message": "no longer consuming from sample-queue"
  }

:query queue: the name of queue
:reqheader Authorization: optional OAuth token to authenticate
:statuscode 200: no error
:statuscode 401: unauthorized request
:statuscode 403: failed to cancel consumer
:statuscode 404: unknown worker
        """
        if not self.is_worker(workername):
            raise web.HTTPError(404, f"Unknown worker '{workername}'")
        queue = self.get_argument('queue')
        logger.info("Canceling consumer '%s' from worker '%s'", queue, workername)
        response = self.capp.control.broadcast('cancel_consumer', arguments={'queue': queue}, destination=[workername], reply=True)
        if response and 'ok' in response[0][workername]:
            self.write(dict(message=response[0][workername]['ok']))
        else:
            logger.error(response)
            self.set_status(403)
            reason = self.error_reason(workername, response)
            self.write(f"Failed to cancel '{queue}' consumer from '{workername}' worker: {reason}")