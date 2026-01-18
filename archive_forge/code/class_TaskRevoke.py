import logging
from tornado import web
from . import BaseApiHandler
class TaskRevoke(ControlHandler):

    @web.authenticated
    def post(self, taskid):
        """
Revoke a task

**Example request**:

.. sourcecode:: http

  POST /api/task/revoke/1480b55c-b8b2-462c-985e-24af3e9158f9?terminate=true
  Content-Length: 0
  Content-Type: application/x-www-form-urlencoded; charset=utf-8
  Host: localhost:5555

**Example response**:

.. sourcecode:: http

  HTTP/1.1 200 OK
  Content-Length: 61
  Content-Type: application/json; charset=UTF-8

  {
      "message": "Revoked '1480b55c-b8b2-462c-985e-24af3e9158f9'"
  }

:query terminate: terminate the task if it is running
:query signal: name of signal to send to process if terminate (default: 'SIGTERM')
:reqheader Authorization: optional OAuth token to authenticate
:statuscode 200: no error
:statuscode 401: unauthorized request
        """
        logger.info("Revoking task '%s'", taskid)
        terminate = self.get_argument('terminate', default=False, type=bool)
        signal = self.get_argument('signal', default='SIGTERM', type=str)
        self.capp.control.revoke(taskid, terminate=terminate, signal=signal)
        self.write(dict(message=f"Revoked '{taskid}'"))