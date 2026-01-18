from designateclient.v2 import base
class ServiceStatusesController(base.V2Controller):

    def list(self, criterion=None, marker=None, limit=None):
        url = self.build_url('/service_statuses', criterion, marker, limit)
        return self._get(url, response_key='service_statuses')

    def get(self, service_status_id):
        url = f'/service_statuses/{service_status_id}'
        return self._get(url)