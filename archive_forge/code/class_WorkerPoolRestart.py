import logging
from tornado import web
from . import BaseApiHandler
class WorkerPoolRestart(ControlHandler):

    @web.authenticated
    def post(self, workername):
        """
Restart worker's pool

**Example request**:

.. sourcecode:: http

  POST /api/worker/pool/restart/celery@worker2 HTTP/1.1
  Content-Length: 0
  Host: localhost:5555

**Example response**:

.. sourcecode:: http

  HTTP/1.1 200 OK
  Content-Length: 56
  Content-Type: application/json; charset=UTF-8

  {
      "message": "Restarting 'celery@worker2' worker's pool"
  }

:reqheader Authorization: optional OAuth token to authenticate
:statuscode 200: no error
:statuscode 401: unauthorized request
:statuscode 403: pool restart is not enabled (see CELERYD_POOL_RESTARTS)
:statuscode 404: unknown worker
        """
        if not self.is_worker(workername):
            raise web.HTTPError(404, f"Unknown worker '{workername}'")
        logger.info("Restarting '%s' worker's pool", workername)
        response = self.capp.control.broadcast('pool_restart', arguments={'reload': False}, destination=[workername], reply=True)
        if response and 'ok' in response[0][workername]:
            self.write(dict(message=f"Restarting '{workername}' worker's pool"))
        else:
            logger.error(response)
            self.set_status(403)
            reason = self.error_reason(workername, response)
            self.write(f"Failed to restart the '{workername}' pool: {reason}")