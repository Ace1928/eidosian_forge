import logging
from tornado import web
from . import BaseApiHandler
class WorkerPoolAutoscale(ControlHandler):

    @web.authenticated
    def post(self, workername):
        """
Autoscale worker pool

**Example request**:

.. sourcecode:: http

  POST /api/worker/pool/autoscale/celery@worker2?min=3&max=10 HTTP/1.1
  Content-Length: 0
  Content-Type: application/x-www-form-urlencoded; charset=utf-8
  Host: localhost:5555

**Example response**:

.. sourcecode:: http

  HTTP/1.1 200 OK
  Content-Length: 66
  Content-Type: application/json; charset=UTF-8

  {
      "message": "Autoscaling 'celery@worker2' worker (min=3, max=10)"
  }

:query min: minimum number of pool processes
:query max: maximum number of pool processes
:reqheader Authorization: optional OAuth token to authenticate
:statuscode 200: no error
:statuscode 401: unauthorized request
:statuscode 403: autoscaling is not enabled (see CELERYD_AUTOSCALER)
:statuscode 404: unknown worker
        """
        if not self.is_worker(workername):
            raise web.HTTPError(404, f"Unknown worker '{workername}'")
        min = self.get_argument('min', type=int)
        max = self.get_argument('max', type=int)
        logger.info("Autoscaling '%s' worker by '%s'", workername, (min, max))
        response = self.capp.control.broadcast('autoscale', arguments={'min': min, 'max': max}, destination=[workername], reply=True)
        if response and 'ok' in response[0][workername]:
            self.write(dict(message=f"Autoscaling '{workername}' worker (min={{min}}, max={{max}})"))
        else:
            logger.error(response)
            self.set_status(403)
            reason = self.error_reason(workername, response)
            self.write(f"Failed to autoscale '{workername}' worker: {reason}")