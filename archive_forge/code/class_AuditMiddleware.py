import copy
import functools
from oslo_config import cfg
from oslo_context import context as oslo_context
from oslo_log import log as logging
from pycadf import cadftaxonomy as taxonomy
from pycadf import cadftype
from pycadf import reason
from pycadf import reporterstep
from pycadf import resource
from pycadf import timestamp
import webob.dec
from keystonemiddleware._common import config
from keystonemiddleware.audit import _api
from keystonemiddleware.audit import _notifier
class AuditMiddleware(object):
    """Create an audit event based on request/response.

    The audit middleware takes in various configuration options such as the
    ability to skip audit of certain requests. The full list of options can
    be discovered here:
    https://docs.openstack.org/keystonemiddleware/latest/audit.html
    """

    def __init__(self, app, **conf):
        self._application = app
        self._conf = config.Config('audit', AUDIT_MIDDLEWARE_GROUP, list_opts(), conf)
        global _LOG
        _LOG = logging.getLogger(conf.get('log_name', __name__))
        self._service_name = conf.get('service_name')
        self._ignore_req_list = [x.upper().strip() for x in conf.get('ignore_req_list', '').split(',')]
        self._cadf_audit = _api.OpenStackAuditApi(conf.get('audit_map_file'), _LOG)
        self._notifier = _notifier.create_notifier(self._conf, _LOG)

    def _create_event(self, req):
        event = self._cadf_audit._create_event(req)
        req.environ['cadf_event'] = event
        return event

    @_log_and_ignore_error
    def _process_request(self, request):
        self._notifier.notify(request.environ['audit.context'], 'audit.http.request', self._create_event(request).as_dict())

    @_log_and_ignore_error
    def _process_response(self, request, response=None):
        if 'cadf_event' not in request.environ:
            self._create_event(request)
        event = request.environ['cadf_event']
        if response:
            if response.status_int >= 200 and response.status_int < 400:
                result = taxonomy.OUTCOME_SUCCESS
            else:
                result = taxonomy.OUTCOME_FAILURE
            event.reason = reason.Reason(reasonType='HTTP', reasonCode=str(response.status_int))
        else:
            result = taxonomy.UNKNOWN
        event.outcome = result
        event.add_reporterstep(reporterstep.Reporterstep(role=cadftype.REPORTER_ROLE_MODIFIER, reporter=resource.Resource(id='target'), reporterTime=timestamp.get_utc_now()))
        self._notifier.notify(request.environ['audit.context'], 'audit.http.response', event.as_dict())

    @webob.dec.wsgify
    def __call__(self, req):
        if req.method in self._ignore_req_list:
            return req.get_response(self._application)
        req.environ['audit.context'] = oslo_context.get_admin_context().to_dict()
        self._process_request(req)
        try:
            response = req.get_response(self._application)
        except Exception:
            self._process_response(req)
            raise
        else:
            self._process_response(req, response)
        return response